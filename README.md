# Emalak invest

![Go Build](https://github.com/u1ug/Emalak-invest/actions/workflows/docker-image.yml/badge.svg)

### Структура проекта
* config - модуль для чтения конфигурационного файла
* moex - модуль для получения данных Московской биржи через <a href="https://pypi.org/project/apimoex/">API</a>
* postgres - модуль для работы с СУБД PostgreSQL
* config.json - файл конфигурации проекта
* Dockerfile - файл для сборки Docker контейнера
* main.py - точка входа Frontend приложения
* markowitz.py - реализация модели Марковица
* optimizer.py - оптимизация портфеля

### Сборка и запуск проекта
git clone https://github.com/u1ug/Emalak-invest
<br>
cd Emalak-invest
<br>
docker build -t emalak .
<br>
docker run -p 28003:28003 emalak

### Альтернативный вариант сборки и запуска
docker pull ghcr.io/u1ug/emalak-invest:latest
<br>
docker run -p 28003:28003 ghcr.io/u1ug/emalak-invest

### Docker контейнер
https://github.com/users/u1ug/packages/container/package/emalak-invest
