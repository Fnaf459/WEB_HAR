Реализовано: Веб-сайт - черный одностроничник с белым текстом, справа снизу поле заполнения CHAT ID, 
по которой потом будет приходить уведомления в телеграм об опасных действиях обнаруженных по камерам в режиме реального времени с названием опасного действия и на какой камере обнаружено. 
Оповещение на самом сайте в виде всплывающего уведомления с названием опасного действия и на какой камере обнаружено. 
Кнопка для добавление камеры(при нажатии выпадает меню с выбором камер среди найденных на устройстве), 
кнопки удаления или изменения выбранной камеры (при нажатии выпадает меню с выбором камер среди найденных на устройстве для замены), 
возможность изменения названия камеры, проверка на добавления камеры (чтобы исключить возможность ещё раз добавить одну и ту же камеру), 
отдельная обработка видеопотока с каждой камеры 
Масштабирование изображения с камер при увеличении количества камер для адекватного просмотра 
(на одном экране может удобно расположиться до 21 камеры, проверено на экране с разрешением 1920х1080 15.6"). 
С помощью готовой предобученной модели глубоких нейронных сетей r2plus1d_18(легковесная) обученная на наборе данных Kinetics-400 (определяет 400 различных действий) 
классифицируются действия человека в режиме реального времени с видеокамер 
Реализованно отслеживание силуэта человека с помощью библиотеки mediapipe сделанное на основе основных точек на теле человека (лицо, суставы) 
Если обнаружено опасное действие, отобранное по логике нарушение правил и норм внутри организаций, которое описано в отдельном .csv файле, 
то прямоугольник обозначающий отслеживание силуэта человека становится красным на самой камере помечается действие и его обнаружение красным, 
уведомления на самом сайте всплывающие с название действия и на какой камере обнаружено, а также такое же увежомление в тг 
Вывод основных характеристик работы модели (ФПС и уверенность (% уверенности модели в соответствии параметров с предсказанной классификацией действия на кадре) модели)

Модель была запущенна на ноутбуке: 15.6" Ноутбук GIGABYTE G5 MF черный 
Экран: Full HD (1920x1080), IPS Процессор: Intel Core i5-12500H, ядра: 4 + 8 х 2.5 ГГц + 1.8 ГГц 
ОЗУ: 24 ГБ 3200Гц 
SSD 512 ГБ 
Видеокарта: GeForce RTX 4050 6 ГБ Laptop

При работе только на CPU: 2 фпс(1 камера), 1 фпс(2 камеры)
При работе на GPU: 6 фпс(1 камера), 4 фпс(2 камеры)
При оптимизации через TensorRT: 10 фпс(1 камера), 7 фпс(2 камеры)
