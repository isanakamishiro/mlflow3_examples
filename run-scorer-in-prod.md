# 本番運用の品質モニタリング (スコアラーの自動実行)

備考

ベータ版

この機能は [ベータ版](/aws/ja/release-notes/release-types)です。

MLflow では、本番運用トレースのサンプルに対してスコアラーを自動的に実行し、品質を継続的に監視することができます。

主な利点:

* 手動による介入のない **自動品質評価**
* カバレッジと計算コストのバランスをとる **ための柔軟なサンプリング**
* 開発から同じスコアラーを使用した **一貫した評価**
* 定期的なバックグラウンド実行による **継続的なモニタリング**

## 前提 条件[​](#前提-条件 "前提-条件 への直接リンク")

1. MLflow と必要なパッケージをインストールする

   Bash

   ```
   pip install --upgrade "mlflow[databricks]>=3.1.0" openai

   ```
2. MLflow エクスペリメントを作成するには、[環境のセットアップに関するクイックスタート](/aws/ja/mlflow3/genai/getting-started/connect-environment)に従ってください。
3. トレースを使用した本番運用アプリケーションのMLflow インストゥルメント化
4. モニタリング出力を保存するための`CREATE TABLE`権限を持つUnity Catalogスキーマへのアクセス。

注記

[Databricks 試用版アカウント](/aws/ja/getting-started/express-setup)を使用している場合は、Unity Catalog スキーマ `workspace.default`に対する CREATE TABLE 権限があります。

## ステップ1:本番運用トレースで採点者をテストする[​](#ステップ1本番運用トレースで採点者をテストする "ステップ1本番運用トレースで採点者をテストする への直接リンク")

まず、本番運用で使用するスコアラーがトレースを評価できるかどうかをテストする必要があります。

ヒント

開発時に`mlflow.genai.evaluate()`で本番運用アプリを`predict_fn`として使用していた場合、スコアラーはすでに互換性がある可能性があります。

警告

MLflow 現在、本番運用 モニタリングのための [事前定義されたスコアラー](/aws/ja/mlflow3/genai/eval-monitor/predefined-judge-scorers) の使用のみをサポートしています。 本番運用でカスタムコードベースまたは Databricksベースのスコアラーを実行する必要がある場合は、 アカウント担当者にお問い合わせください。LLM

1. `mlflow.genai.evaluate()`を使用して、トレースのサンプルでスコアラーをテストします

   Python

   ```
   import mlflow

   from mlflow.genai.scorers import (
       Guidelines,
       RelevanceToQuery,
       RetrievalGroundedness,
       RetrievalRelevance,
       Safety,
   )

   # Get a sample of up to 10 traces from your experiment
   traces = mlflow.search_traces(max_results=10)

   # Run evaluation to test the scorers
   mlflow.genai.evaluate(
       data=traces,
       scorers=[
           RelevanceToQuery(),
           RetrievalGroundedness(),
           RetrievalRelevance(),
           Safety(),
           Guidelines(
               name="mlflow_only",
               # Guidelines can refer to the request and response.
               guidelines="If the request is unrelated to MLflow, the response must refuse to answer.",
           ),
           # You can have any number of guidelines.
           Guidelines(
               name="customer_service_tone",
               guidelines="""The response must maintain our brand voice which is:
       - Professional yet warm and conversational (avoid corporate jargon)
       - Empathetic, acknowledging emotional context before jumping to solutions
       - Proactive in offering help without being pushy

       Specifically:
       - If the customer expresses frustration, anger, or disappointment, the first sentence must acknowledge their emotion
       - The response must use "I" statements to take ownership (e.g., "I understand" not "We understand")
       - The response must avoid phrases that minimize concerns like "simply", "just", or "obviously"
       - The response must end with a specific next step or open-ended offer to help, not generic closings""",
           ),
       ],
   )

   ```
2. MLflow トレース UI を使用して、実行されたスコアラーを確認する

   この場合、 `RetrievalGroundedness()` と `RetrievalRelevance()` のスコアラーを実行したにもかかわらず、MLflow UI に表示されないことがわかります。これは、これらのスコアラーがトレースを操作しないことを示しており、次のステップで有効にすべきではありません。

## ステップ 2: モニタリングを有効にする[​](#ステップ-2-モニタリングを有効にする "ステップ-2-モニタリングを有効にする への直接リンク")

それでは、モニタリングサービスを有効にしましょう。 有効にすると、モニタリング サービスは、評価されたトレースのコピーをMLflow エクスペリメントから、指定した スキーマのDelta Unity Catalogテーブルに同期します。

important

一度設定すると、Unity Catalog スキーマは変更できません。

* Using the UI* Using the SDK

以下の記録に従って、UI を使用して、手順 1 で正常に実行されたスコアラーを有効にします。サンプリングレートを選択すると、その割合のトレースでのみスコアラーが実行されます(たとえば、「 `1.0` を入力すると、トレースの100%でスコアラーが実行され、 `.2` は20%で実行されます)。

スコアラーごとのサンプリング レートを設定する場合は、SDK を使用する必要があります。

![trace](https://assets.docs.databricks.com/_static/images/mlflow3-genai/new-images/enable-monitor.gif)

次のコード スニペットを使用して、手順 1 で正常に実行されたスコアラーを有効にします。サンプリングレートを選択すると、その割合のトレースでのみスコアラーが実行されます(たとえば、「 `1.0` を入力すると、トレースの100%でスコアラーが実行され、 `.2` は20%で実行されます)。 オプションで、スコアラーごとのサンプリングレートを設定できます。

Python

```
# These packages are automatically installed with mlflow[databricks]
from databricks.agents.monitoring import create_external_monitor, AssessmentsSuiteConfig, BuiltinJudge, GuidelinesJudge

external_monitor = create_external_monitor(
    # Change to a Unity Catalog schema where you have CREATE TABLE permissions.
    catalog_name="workspace",
    schema_name="default",
    assessments_config=AssessmentsSuiteConfig(
        sample=1.0,  # sampling rate
        assessments=[
            # Predefined scorers "safety", "groundedness", "relevance_to_query", "chunk_relevance"
            BuiltinJudge(name="safety"),  # or {'name': 'safety'}
            BuiltinJudge(
                name="groundedness", sample_rate=0.4
            ),  # or {'name': 'groundedness', 'sample_rate': 0.4}
            BuiltinJudge(
                name="relevance_to_query"
            ),  # or {'name': 'relevance_to_query'}
            BuiltinJudge(name="chunk_relevance"),  # or {'name': 'chunk_relevance'}
            # Guidelines can refer to the request and response.
            GuidelinesJudge(
                guidelines={
                    # You can have any number of guidelines, each defined as a key-value pair.
                    "mlflow_only": [
                        "If the request is unrelated to MLflow, the response must refuse to answer."
                    ],  # Must be an array of strings
                    "customer_service_tone": [
                        """The response must maintain our brand voice which is:
    - Professional yet warm and conversational (avoid corporate jargon)
    - Empathetic, acknowledging emotional context before jumping to solutions
    - Proactive in offering help without being pushy

    Specifically:
    - If the customer expresses frustration, anger, or disappointment, the first sentence must acknowledge their emotion
    - The response must use "I" statements to take ownership (e.g., "I understand" not "We understand")
    - The response must avoid phrases that minimize concerns like "simply", "just", or "obviously"
    - The response must end with a specific next step or open-ended offer to help, not generic closings"""
                    ],
                }
            ),
        ],
    ),
)

print(external_monitor)

```

## ステップ3.モニターの更新[​](#ステップ3モニターの更新 "ステップ3モニターの更新 への直接リンク")

スコアラーの設定を変更するには、 `update_external_monitor()`を使用します。設定はステートレスです - つまり、更新によって完全に上書きされます。変更する既存の設定を取得するには、 `get_external_monitor()`を使用します。

* Using the UI* Using the SDK

以下の録画に従って、UIを使用してスコアラーを更新してください。

![trace](https://assets.docs.databricks.com/_static/images/mlflow3-genai/new-images/update-monitor.gif)

Python

```
# These packages are automatically installed with mlflow[databricks]
from databricks.agents.monitoring import update_external_monitor, get_external_monitor
import os

config = get_external_monitor(experiment_id=os.environ["MLFLOW_EXPERIMENT_ID"])
print(config)

external_monitor = update_external_monitor(
    # You must pass the experiment_id of the experiment you want to update.
    experiment_id=os.environ["MLFLOW_EXPERIMENT_ID"],
    # Change to a Unity Catalog schema where you have CREATE TABLE permissions.
    assessments_config=AssessmentsSuiteConfig(
        sample=1.0,  # sampling rate
        assessments=[
            # Predefined scorers "safety", "groundedness", "relevance_to_query", "chunk_relevance"
            BuiltinJudge(name="safety"),  # or {'name': 'safety'}
            BuiltinJudge(
                name="groundedness", sample_rate=0.4
            ),  # or {'name': 'groundedness', 'sample_rate': 0.4}
            BuiltinJudge(
                name="relevance_to_query"
            ),  # or {'name': 'relevance_to_query'}
            BuiltinJudge(name="chunk_relevance"),  # or {'name': 'chunk_relevance'}
            # Guidelines can refer to the request and response.
            GuidelinesJudge(
                guidelines={
                    # You can have any number of guidelines, each defined as a key-value pair.
                    "mlflow_only": [
                        "If the request is unrelated to MLflow, the response must refuse to answer."
                    ],  # Must be an array of strings
                    "customer_service_tone": [
                        """The response must maintain our brand voice which is:
    - Professional yet warm and conversational (avoid corporate jargon)
    - Empathetic, acknowledging emotional context before jumping to solutions
    - Proactive in offering help without being pushy

    Specifically:
    - If the customer expresses frustration, anger, or disappointment, the first sentence must acknowledge their emotion
    - The response must use "I" statements to take ownership (e.g., "I understand" not "We understand")
    - The response must avoid phrases that minimize concerns like "simply", "just", or "obviously"
    - The response must end with a specific next step or open-ended offer to help, not generic closings"""
                    ],
                }
            ),
        ],
    ),
)

print(external_monitor)

```

## ステップ4.モニタリング結果の使用[​](#ステップ4モニタリング結果の使用 "ステップ4モニタリング結果の使用 への直接リンク")

モニタリング ジョブを初めて実行するには、~15 分から 30 分かかります。 最初の実行後、15 分ごとに実行されます。 本番運用のトラフィックが大量にある場合、ジョブの完了にさらに時間がかかる可能性があることに注意してください。

ジョブが実行されるたびに、次のことが行われます。

1. トレースのサンプルに対して各スコアラーを実行します

   * スコアラーごとにサンプリングレートが異なる場合、モニタリングジョブは、同じトレースをできるだけ多くスコアリングしようとします。たとえば、スコアラー A のサンプリング レートが 20%で、スコアラー B のサンプリング レートが 40% の場合、トレースの同じ 20% が A と B に使用されます。
2. スコアラーからの [フィードバック](/aws/ja/mlflow3/genai/tracing/data-model#feedback) を MLflow エクスペリメントの各トレースに添付します
3. すべてのトレース (サンプリングされたトレースだけでなく) のコピーを、手順 1 で構成された Delta テーブルに書き込みます。

モニタリング結果は、 MLflow エクスペリメントのTraceタブを使用して表示できます。または、生成された Delta テーブルで SQL または Spark を使用してトレースのクエリを実行することもできます。

## 次のステップ[​](#次のステップ "次のステップ への直接リンク")

これらの推奨アクションとチュートリアルで旅を続けてください。

* [本番運用トレースを使用してアプリの品質を向上させる](/aws/ja/mlflow3/genai/eval-monitor/evaluate-app) - LLM を使用してセマンティック評価を作成する
* [評価データセットの構築](/aws/ja/mlflow3/genai/eval-monitor/build-eval-dataset) - モニタリングの結果を使用して、パフォーマンスの低いトレースを評価データセットにキュレーションし、品質を向上させます。

## リファレンスガイド[​](#リファレンスガイド "リファレンスガイド への直接リンク")

このガイドで説明されている概念と機能の詳細なドキュメントをご覧ください。

* [本番運用 モニタリング](/aws/ja/mlflow3/genai/eval-monitor/concepts/production-monitoring) - 本番運用 モニタリング SDK の詳細

**2025年5月18日**に最終更新