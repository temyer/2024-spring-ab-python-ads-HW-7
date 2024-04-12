# 2024-spring-ab-python-ads-HW-7

На Kind-кластере развернуть `SeldonCore` + `Prometheus` + `Grafana`. Реализовать с помощью `MLServer` и `SeldonDeployment` приложение на основе uplift-модели с endpoint'ом `/predict`, который по фичам клиента возвращает предсказание uplift (вероятность покупки с промо - вероятность покупки без промо). Построить в `Grafana` дэшборд, показывающий количество успешных реквестов во времени.

Все необходимые шаги описать в `Makefile` вида:

```yaml
train: python train.py

build: mlserver build . -t [YOUR_CONTAINER_REGISTRY]/[IMAGE_NAME]

kind-cluster: kind create cluster --name seldon-cluster --config kind-cluster.yaml --image=kindest/node:v1.21.1

ambassador:
  ###

seldon-core:
  ###
```

Все необходимые `yaml`-файлы (например, `SeldonDeployment` и `Podmonitor` для `Prometheus`), а также скрин дэшборда в `Graphana` также должны быть в репозитории.

**Критерии**

1. Подготовлены модули для обучения, инференса и выполения тестовых реквестов - +2
2. В `Makefile` есть все необходимые шаги для сборки образа в `MLServer` - +2
3. В `Makefile` есть все необходимые шаги для деплоя необходимых компонентов (включая `port forwarding`) - +4
4. В `Makefile` есть шаг для тестовых реквестов - +1
5. Прикреплен скрин дэшборда в `Grafana` - +1

### Скриншот из интерфейса graphana

![Alt text](./Screenshot_2024-04-13_02-55-28.png?raw=true "Grahpana")
