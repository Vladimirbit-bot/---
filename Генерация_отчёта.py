    from gigachat import GigaChat    
    from gigachat.models import Chat, Messages, MessagesRole    
         
         
    def generate_walk_report(walk_data: dict, gigachat_credentials: str) -> str:    
        """    
        Генерация текстового отчёта о прогулке с помощью GigaChat.    
         
        Args:    
            walk_data: словарь с данными о прогулке    
            gigachat_credentials: авторизационные данные GigaChat API    
         
        Returns:    
            str: сгенерированный отчёт    
        """    
        prompt = f"""    
        Сгенерируй дружелюбный отчёт о прогулке собаки для владельца.    
         
        Данные прогулки:    
        - Кличка собаки: {walk_data['dog_name']}    
        - Длительность: {walk_data['duration_minutes']} минут    
        - Пройденное расстояние: {walk_data['distance_km']} км    
        - Средняя скорость: {walk_data['avg_speed']} км/ч    
        - Количество остановок: {walk_data['stops_count']}    
        - Обнаруженные события: {', '.join(walk_data['events'])}    
        - Уровень активности: {walk_data['activity_level']}    
        - Погода: {walk_data['weather']}    
         
        Сделай отчёт тёплым и информативным, как будто пишешь заботливому хозяину.    
        """    
         
        with GigaChat(credentials=gigachat_credentials, verify_ssl_certs=False) as giga:    
            response = giga.chat(    
                Chat(    
                    messages=[    
                        Messages(    
                            role=MessagesRole.SYSTEM,    
                            content="Ты помощник сервиса по выгулу собак. Пишешь отчёты для владельцев."    
                        ),    
                        Messages(    
                            role=MessagesRole.USER,    
                            content=prompt    
                        )    
                    ],    
                    temperature=0.7,    
                    max_tokens=500    
                )    
            )    
         
        return response.choices[0].message.content    
         
         
    # Пример использования    
    walk_data = {    
        'dog_name': 'Барон',    
        'duration_minutes': 45,    
        'distance_km': 2.3,    
        'avg_speed': 3.1,    
        'stops_count': 5,    
        'events': ['встреча с другой собакой', 'короткий отдых у скамейки'],    
        'activity_level': 'высокий',    
        'weather': 'солнечно, +18°C'    
    }    
         
    # Авторизационные данные берутся из личного кабинета GigaChat API    
    GIGACHAT_CREDENTIALS = "ВАШ_АВТОРИЗАЦИОННЫЙ_КЛЮЧ"    
         
    report = generate_walk_report(walk_data, GIGACHAT_CREDENTIALS)    
    print(report)    
         
