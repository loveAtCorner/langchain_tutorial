import requests

# 替换为你的API密钥
API_KEY = 'e942c23175702a30a5533a249215b4f2'

def get_weather(city):
    url = f'http://apis.juhe.cn/simpleWeather/query'
    params = {
        'city': city,
        'key': API_KEY
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if data['error_code'] == 0:
            result = data['result']
            realtime = result['realtime']
            future = result['future']
            
            print(f"当前天气: {realtime['info']}, 温度: {realtime['temperature']}°C, 湿度: {realtime['humidity']}%")
            print("未来天气预报:")
            for day in future:
                print(f"{day['date']}: {day['weather']}, 温度: {day['temperature']}")
        else:
            print(f"请求失败: {data['reason']}")
    else:
        print(f"HTTP请求失败: {response.status_code}")

if __name__ == "__main__":
    city = input("请输入要查询的城市名称: ")
    get_weather(city)
