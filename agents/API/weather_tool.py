import requests
from langchain_core.tools import BaseTool

# 替换为你的API密钥
API_KEY = 'e942c23175702a30a5533a249215b4f2'


class WeatherTool(BaseTool):
    name = "WeatherQuery"
    description = "Use this tool to query the weather in a specific location."

    def _run(self, location: str):
        url = 'http://apis.juhe.cn/simpleWeather/query'
        params = {
            'city': location,
            'key': API_KEY
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data['error_code'] == 0:
                result = data['result']
                realtime = result['realtime']
                future = result['future']

                # Format current weather and future forecast
                current_weather = (
                    f"当前天气: {realtime['info']}, 温度: {realtime['temperature']}°C, "
                    f"湿度: {realtime['humidity']}%"
                )
                future_forecast = "\n未来天气预报:\n"
                for day in future:
                    future_forecast += f"{day['date']}: {day['weather']}, 温度: {day['temperature']}\n"

                return current_weather + "\n" + future_forecast
            else:
                return f"请求失败: {data['reason']}"
        else:
            return f"HTTP请求失败: {response.status_code}"

    # Optional: you can override _arun if you want to support async execution
    async def _arun(self, location: str):
        raise NotImplementedError("This tool does not support async execution.")


# Example of how to integrate this tool in an agent
if __name__ == "__main__":
    tool = WeatherTool()
    location = input("请输入要查询的城市名称: ")
    result = tool._run(location)  # Use _run instead of run
    print(result)