# MNIST distributed training over BOINC

## Архитектура стенда
- **Сервисы**: `mysql`, `apache` (BOINC server + daemons), два клиента `client1/2` на образе `boinc/mnist-client` с Python/numpy, общий `project` volume, `results` volume для ассимиляции.
- **Приложение**: BOINC app `mnist` (версия 1.3, `app_version_num=103`) – чистый NumPy‑логистический классификатор, скачивает `mnist.npz`, сэмплирует подмножество, обучает Softmax на 10 классов. Исполняемый скрипт: `apps/mnist/1.3/x86_64-pc-linux-gnu/mnist.py`.
- **Шаблоны**: `templates/mnist_in` (один вход `job.json`), `templates/mnist_out` (один выход `metrics.json`).
- **Пайплайн**: `run-mnist.ps1` → укладывает `job.json` в download → `bin/create_work` → `client*` забирают задание → считаются метрики → ассимилятор `apps/mnist/assimilate_mnist.sh` складывает результаты в `/results/mnist/<wu>/metrics.json`.
- **Админка/OPS**: Basic auth восстановлен (`admin`/`zJiQQ3OoIfehM`); проектная конфигурация и демоны подхватываются из `overrides/config.xml` и `overrides/project.inc`.

## Как запускали эксперимент
- Параметры серии: 5 раундов, `sample_count=8000`, `train_epochs=10`, `batch_size=256`, `learning_rate=0.05`, `min_quorum=2`, `target_nresults=2`, `PollSeconds=600`.
- Команда:  
  ```powershell
  cd boinc-server-docker
  powershell -ExecutionPolicy Bypass -File .\run-mnist.ps1 `
    -Rounds 5 -SampleCount 8000 -TrainEpochs 10 -BatchSize 256 -LearningRate 0.05 -PollSeconds 600
  ```
- Перед запуском: обновили версию приложения (`apps/mnist/1.3/...`), прогнали `bin/update_versions --appname mnist`, сбросили клиентов `boinccmd --project ... reset`, чтобы они скачали новую версию с логированием лосса.

## Результаты BOINC-прогона (distributed)
Файлы: `results/mnist/mnist_run20251216024339_r*_0/metrics.json`.

| round | seed  | client  | train_acc | val_acc | train_loss | val_loss | runtime_s |
|-------|-------|---------|-----------|---------|------------|----------|-----------|
| 1 | 714562 | client1 | 0.8748 | 0.8531 | 0.5508 | 0.5843 | 0.87 |
| 2 | 910552 | client2 | 0.8688 | 0.8600 | 0.5616 | 0.5919 | 0.89 |
| 3 | 625989 | client1 | 0.8688 | 0.8569 | 0.5660 | 0.5881 | 0.86 |
| 4 | 962826 | client2 | 0.8723 | 0.8613 | 0.5599 | 0.5641 | 0.87 |
| 5 | 103343 | client1 | 0.8661 | 0.8913 | 0.5557 | 0.5385 | 0.85 |

Сводно: средняя `val_acc = 0.8645 ± 0.0137`, средняя `val_loss = 0.5734 ± 0.0199`. Валид. метрика выросла монотонно по эпохам в каждом раунде; лосс спадал стабильно (см. графики ниже).

## Локальный референс
- Тем же кодом (`apps/mnist/1.3/.../mnist.py`) прогнаны локальные job’ы с идентичными payload’ами и seed’ами из distributed ранa.
- Файлы: `results_local/*_local_metrics.json`.
- Итоговые `val_acc/val_loss` совпали с распределёнными (детерминированный NumPy и одинаковые seed’ы), разброс аналогичный: `val_acc = 0.8645 ± 0.0137`, `val_loss = 0.5734 ± 0.0199`.

## Визуализации
- Средние кривые точности/лосса (с диапазоном min–max по пяти раундам) и локальный средний референс: `reports/mnist/acc_loss_curves.png`.
- Финальные валид. точности по каждому раунду, сравнение distributed vs local: `reports/mnist/final_val_acc.png`.

## Ключевые наблюдения
- Обучение на 8k сэмплов и 10 эпох сходится быстро: валид. лосс падает с ~1.45 до ~0.55, точность растёт с ~0.75–0.80 до ~0.86–0.89 за 10 эпох.
- Разброс между раундами небольшой; разброс в `val_acc` ~±1.4 п.п. зависит от seed (выборки из MNIST).
- Распределённый прогон эквивалентен локальному по качеству и времени (каждая задача ~0.85–0.9 с CPU) — хорошая проверка цепочки BOINC/ассимилятора, а не масштабирования по скорости.

## Что пришлось чинить по пути
- **OPS авторизация**: в секрете `/run/secrets/html/ops/.htpasswd` не было пароля — пересоздали htpasswd (admin/zJiQQ3OoIfehM), символическая ссылка в `html/ops/.htpasswd` восстановлена.
- **Пустые загрузки**: старый `mnist_in` генерировал путь `download/download/...` → `md5_file fopen() failed`. Починили шаблон (file_number/ copy_file) и команду `create_work` в `run-mnist.ps1` (явные input/output templates, прямое имя файла без `download/` префикса).
- **Клиенты не брали работу**: в compose был невалидный authenticator, scheduler возвращал `Bad authenticator`. Перепривязали clients на актуальный ключ `8349f3485acdb83aa9dd1cf32d7038be` и добавили автовызов `boinccmd --project ... update` в цикл ожидания.
- **Обновление версии приложения**: для логирования лосса добавили версию `mnist` 1.3 (app_version_num=103) и после `update_versions` сбросили клиентов (`boinccmd --project ... reset`), чтобы они скачали новый `mnist.py` (иначе оставался старый файл на клиентах).

## Как воспроизвести сейчас
1) Убедиться, что стек поднят: `docker compose up -d`.  
2) (если правили app) в контейнере: `docker compose exec apache bash -lc "cd /home/boincadm/project && yes | bin/update_versions --appname mnist --noconfirm"` и `boinccmd --project ... reset` на клиентах.  
3) Запустить серию:  
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\run-mnist.ps1 `
     -Rounds 5 -SampleCount 8000 -TrainEpochs 10 -BatchSize 256 -LearningRate 0.05 -PollSeconds 600
   ```  
4) Результаты: `results/mnist/mnist_run*/metrics.json`, графики в `reports/mnist/`.  
5) OPS панель: http://localhost:8082/boincserver_ops/ (admin/zJiQQ3OoIfehM).

## Выводы и идеи дальше
- Цепочка BOINC → validator → script_assimilator теперь устойчива: задания валидируются/ассимилируются за ~5–6 секунд с двумя клиентами.
- Добавлен учёт лосса и истории обучения в метриках, что делает сравнение distributed vs local прозрачным.
- Следующие шаги для «курсовой» полноты: увеличить `sample_count`/эпохи до полного датасета, добавить шум/аугментации и замерять влияние количества клиентов на wall-clock и вариативность, а также добавить выгрузку весов для анализа распределения коэффициентов.
