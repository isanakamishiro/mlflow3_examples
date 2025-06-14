import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from mlflow.entities import SpanType
from typing import Generator, Any
from databricks_langchain import ChatDatabricks, UCFunctionToolkit
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    ToolMessage,
    AIMessageChunk,
)
from langchain_core.tools import BaseTool, tool
from functools import reduce


# ダミーツール
@tool
def get_weather(city: str) -> str:
    """指定された都市の天気を取得します

    Args:
        city (str): 都市名

    Returns:
        str: 天気情報
    """
    return f"It's always sunny in {city}!"


# Tracingの有効化
mlflow.langchain.autolog()

# Agent
class SimpleResponsesAgent(ResponsesAgent):
    def __init__(self, model, tools: list[BaseTool]):
        """SimpleResponsesAgentの初期化

        Args:
            model: 使用するモデル
            tools (list[BaseTool]): 使用するツールのリスト
        """
        self.model = model
        self.tools = tools

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """リクエストに基づいて予測を行います

        Args:
            request (ResponsesAgentRequest): 予測リクエスト

        Returns:
            ResponsesAgentResponse: 予測結果のレスポンス
        """
        events = [
            event
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        outputs = [event.item for event in events]
        # usage総量を計算
        usages = [event.usage for event in events]
        total_usage = {
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
            **reduce(
                lambda x, y: {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)},
                usages,
            ),
        }

        return ResponsesAgentResponse(output=outputs, usage=total_usage)

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """ストリームモードで予測を行います

        Args:
            request (ResponsesAgentRequest): 予測リクエスト

        Yields:
            ResponsesAgentStreamEvent: ストリームイベント
        """
        messages, params = self._convert_request_to_lc_request(request)
        react_agent = create_react_agent(self.model.bind(**params), tools=self.tools)

        for chunk in react_agent.stream({"messages": messages}, stream_mode="updates"):
            for value in chunk.values():
                messages = value.get("messages", [])
                responses = self._convert_lc_messages_to_response(messages)
                for response in responses:
                    yield response

    @mlflow.trace(span_type=SpanType.PARSER)
    def _convert_request_to_lc_request(
        self, request: ResponsesAgentRequest
    ) -> (list[BaseMessage], dict[str, Any]):
        """リクエストをLangChainのメッセージおよびパラメータ形式に変換します

        Args:
            request (ResponsesAgentRequest): 変換するリクエスト

        Returns:
            tuple: メッセージリスト、パラメータ辞書
        """

        lc_request = request.model_dump_compat(exclude_none=True)
        custom_inputs = lc_request.pop("custom_inputs", {})

        # custom_inputsは通常のパラメータとして展開
        lc_request.update(custom_inputs)
        messages = lc_request.pop("input")

        # LangChainで有効なパラメータのみに限定
        valid_params = [
            "temperature",
            "max_output_tokens",
            "top_p",
            "top_k",
        ]
        params = {k: v for k, v in lc_request.items() if k in valid_params}
        if "max_output_tokens" in params:
            params["max_tokens"] = params.pop("max_output_tokens")

        return messages, params

    @mlflow.trace(span_type=SpanType.PARSER)
    def _convert_lc_messages_to_response(
        self, messages: list[BaseMessage]
    ) -> list[ResponsesAgentStreamEvent]:
        """LangChainメッセージをレスポンス出力に変換します

        Args:
            messages (list[BaseMessage]): 変換するメッセージリスト

        Returns:
            list[ResponsesAgentStreamEvent]: レスポンス出力のリスト
        """

        def _create_response_agent_stream_event(
            item, usage, metadata
        ) -> ResponsesAgentStreamEvent:
            return ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=item,
                usage=_convert_lc_usage_to_openai_usage(usage),
                metadata=metadata,
            )

        def _convert_lc_usage_to_openai_usage(usage: dict[str, int]) -> dict[str, int]:
            return {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        outputs = []
        for message in messages:
            if isinstance(message, ToolMessage):
                item = self.create_function_call_output_item(
                    output=message.content,
                    call_id=message.tool_call_id,
                )
                item["content"] = item["output"] # 互換性のため
                metadata = message.response_metadata
                usage = metadata.pop("usage", {})
                outputs.append(
                    _create_response_agent_stream_event(item, usage, metadata)
                )
            elif (
                isinstance(message, (AIMessage, AIMessageChunk)) and message.tool_calls
            ):
                metadata = message.response_metadata
                usage = metadata.pop("usage", {})
                for tool_call in message.tool_calls:
                    item = self.create_function_call_item(
                        id=message.id,
                        call_id=tool_call.get("id"),
                        name=tool_call.get("name"),
                        arguments=str(tool_call.get("args")),
                    )
                    item["content"] = item["name"] # 互換性のため                    
                    outputs.append(
                        _create_response_agent_stream_event(item, usage, metadata)
                    )
                    # 1件目のみusageを設定
                    usage = {}
            elif isinstance(message, (AIMessage, AIMessageChunk)):
                item = self.create_text_output_item(
                    text=message.content,
                    id=message.id,
                )
                metadata = message.response_metadata
                usage = metadata.pop("usage", {})
                outputs.append(
                    _create_response_agent_stream_event(item, usage, metadata)
                )
            else:
                raise ValueError(f"Unknown message: {message}")
        return outputs


# Databricksネイティブのllama-3-1-405b-instructを利用
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-1-405b-instruct"
llm = ChatDatabricks(model=LLM_ENDPOINT_NAME)

# 利用可能なツールとして、get_weather関数とUnity Catalogのsystem.ai配下の関数を設定
func_name = f"system.ai.*"
uc_toolkit = UCFunctionToolkit(function_names=[func_name])
LC_TOOLS = [get_weather] + uc_toolkit.tools

# mlflowにエージェントを設定
agent = SimpleResponsesAgent(model=llm, tools=LC_TOOLS)
mlflow.models.set_model(agent)
