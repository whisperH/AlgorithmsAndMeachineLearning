# 根据图中定义天气之间状态转移概率(transition probability)
Weather2Weather_Prob = {
    'sunny': {
        'sunny': 0.8,
        'rain': 0.2
    },
    'rain': {
        'sunny': 0.4,
        'rain': 0.6
    }
}
# 定义发射概率(emission probability)

Mood2Weather_Prob = {
    'happy': {
        'sunny': 0.8,
        'rain': 0.4,
    },
    'grumpy': {
        'sunny': 0.2,
        'rain': 0.6
    }

}

weather_state = ['sunny', 'rain']
mood_state = ['happy', 'grumpy']

# 初始化两种天气的概率
sunny_init_prob = 2/3
rain_init_prob = 1/3

def predictMood(BobMood):
    weather = {}
    weather_prob = {}
    # 定义概率矩阵
    for iday in range(len(BobMood)+1):
        weather_prob[iday] = {
            'sunny': 0,
            'rain': 0
        }
    weather_prob[0] = {
        'sunny': sunny_init_prob,
        'rain': rain_init_prob
    }
    # 根据Bob的心情，计算每一天可能的概率
    for iday, iday_mood in enumerate(BobMood):
        # 考虑当前天气的状态数量
        for icurrent_weather_state in weather_state:
            weather_prob[iday][icurrent_weather_state] *= Mood2Weather_Prob[iday_mood][icurrent_weather_state]
            # 考虑状态转移之后的概率
            for ifuture_weather_state in weather_state:
                weather_prob[iday+1][ifuture_weather_state] = max(
                    weather_prob[iday][icurrent_weather_state] * Weather2Weather_Prob[icurrent_weather_state][ifuture_weather_state],
                    weather_prob[iday + 1][ifuture_weather_state]
                )

        weather[iday] = 'sunny' if weather_prob[iday]['sunny'] > weather_prob[iday]['rain'] else 'rain'

    return weather, weather_prob

if __name__ == '__main__':


    BobMood = ['happy', 'happy', 'grumpy', 'grumpy', 'grumpy', 'happy']

    weather, weather_prob = predictMood(BobMood)
    print('weather is:')
    print(weather)

    print('probabilities are:')
    print(weather_prob)