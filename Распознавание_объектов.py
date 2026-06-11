from ultralytics import YOLO    
    import cv2    
    import numpy as np    
         
    # Загрузка предобученной модели YOLOv8    
    model = YOLO('yolov8n.pt')  # nano-версия для быстрого inference    
         
    # Классы объектов, которые нас интересуют    
    THREAT_CLASSES = {    
        16: 'dog',      # Другие собаки    
        0: 'person',    # Люди    
        2: 'car',       # Автомобили    
        7: 'truck',     # Грузовики    
        5: 'bus',       # Автобусы    
        1: 'bicycle',   # Велосипеды    
        3: 'motorcycle' # Мотоциклы    
    }    
         
    def detect_threats(frame):    
        """    
        Распознавание объектов на кадре и определение угроз.    
         
        Args:    
            frame: numpy array, изображение с камеры робота    
         
        Returns:    
            threats: список обнаруженных угроз с координатами    
        """    
        # Выполнение inference    
        results = model(frame, verbose=False)    
         
        threats = []    
         
        for result in results:    
            boxes = result.boxes    
         
            for box in boxes:    
                cls_id = int(box.cls[0])    
                conf = float(box.conf[0])    
         
                # Фильтрация по интересующим классам    
                if cls_id in THREAT_CLASSES and conf > 0.5:    
                    x1, y1, x2, y2 = box.xyxy[0].tolist()    
         
                    threat = {    
                        'class': THREAT_CLASSES[cls_id],    
                        'confidence': conf,    
                        'bbox': [x1, y1, x2, y2],    
                        'center': [(x1 + x2) / 2, (y1 + y2) / 2]    
                    }    
                    threats.append(threat)    
         
        return threats    
         
         
    def calculate_threat_level(threats, frame_width, frame_height):    
        """    
        Расчёт уровня угрозы на основе обнаруженных объектов.    
         
        Args:    
            threats: список обнаруженных угроз    
            frame_width: ширина кадра    
            frame_height: высота кадра    
         
        Returns:    
            threat_level: строка ('low', 'medium', 'high', 'critical')    
            actions: рекомендуемые действия    
        """    
        if not threats:    
            return 'low', []    
         
        actions = []    
        max_level = 0    
         
        for threat in threats:    
            x1, y1, x2, y2 = threat['bbox']    
            obj_width = x2 - x1    
            obj_height = y2 - y1    
         
            # Расчёт относительного размера объекта    
            relative_size = (obj_width * obj_height) / (frame_width * frame_height)    
         
            # Расчёт близости к центру кадра (траектории движения)    
            center_x = threat['center'][0]    
            distance_from_center = abs(center_x - frame_width / 2) / (frame_width / 2)    
         
            # Определение уровня угрозы для объекта    
            if threat['class'] == 'dog':    
                # Другие собаки — потенциально высокая угроза    
                if relative_size > 0.1:    
                    max_level = max(max_level, 3)    
                    actions.append('Обнаружена собака вблизи — снизить скорость')    
                elif relative_size > 0.05:    
                    max_level = max(max_level, 2)    
                    actions.append('Обнаружена собака — наблюдение')    
         
            elif threat['class'] in ['car', 'truck', 'bus', 'motorcycle']:    
                # Транспорт — критическая угроза при близком расположении    
                if relative_size > 0.15 and distance_from_center < 0.3:    
                    max_level = max(max_level, 4)    
                    actions.append('КРИТИЧНО: Транспорт на траектории — остановка')    
                elif relative_size > 0.08:    
                    max_level = max(max_level, 3)    
                    actions.append('Транспорт вблизи — изменить маршрут')    
         
            elif threat['class'] == 'person':    
                # Люди — низкая угроза, но требуют внимания    
                if relative_size > 0.2:    
                    max_level = max(max_level, 1)    
                    actions.append('Человек на пути — обойти')    
         
        level_map = {0: 'low', 1: 'low', 2: 'medium', 3: 'high', 4: 'critical'}    
         
        return level_map.get(max_level, 'low'), actions    
         
         
    def process_camera_feed():    
        """Обработка видеопотока с камеры робота."""    
        cap = cv2.VideoCapture(0)  # Или путь к видеофайлу    
         
        while True:    
            ret, frame = cap.read()    
            if not ret:    
                break    
         
            # Распознавание угроз    
            threats = detect_threats(frame)    
         
            # Расчёт уровня угрозы    
            h, w = frame.shape[:2]    
            threat_level, actions = calculate_threat_level(threats, w, h)    
         
            # Вывод информации    
            print(f"Уровень угрозы: {threat_level}")    
            for action in actions:    
                print(f"  → {action}")    
         
            # Визуализация (для отладки)    
            for threat in threats:    
                x1, y1, x2, y2 = [int(x) for x in threat['bbox']]    
                if threat_level == 'low':    
                    color = (0, 255, 0)    
                elif threat_level == 'medium':    
                    color = (0, 165, 255)    
                else:    
                    color = (0, 0, 255)    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)    
                cv2.putText(    
                    frame,    
                    f"{threat['class']} {threat['confidence']:.2f}",    
                    (x1, y1 - 10),    
                    cv2.FONT_HERSHEY_SIMPLEX,    
                    0.5,    
                    color,    
                    2    
                )    
         
            cv2.imshow('Threat Detection', frame)    
         
            if cv2.waitKey(1) & 0xFF == ord('q'):    
                break    
         
        cap.release()    
        cv2.destroyAllWindows()    
         
         
    if __name__ == "__main__":    
        process_camera_feed()   
