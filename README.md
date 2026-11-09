# Генетические алгоритмы и эвристики для университетского расписания (ITC-2007)

Код и результаты к статье «Оптимизация университетского расписания занятий: сравнение генетических алгоритмов и эвристических подходов» (Е. А. Борохов, ИММКН им. И. И. Воровича ЮФУ).

Задача — Curriculum-Based Course Timetabling в официальной постановке ITC-2007 (UD2): жёсткие ограничения H1–H4 и мягкие S1 (вместимость), S2 (минимум учебных дней, ×5), S3 («окна» групп, ×2), S4 (смена аудиторий). Сравниваются шесть методов на задачах comp01–comp07 при бюджете 60 с процессорного времени, 3 запуска.

## Результаты (мягкий штраф, среднее по 3 запускам; * — есть недопустимые запуски)

| Задача | Greedy-Rand | Greedy-Sat | GA | MA | HC | SA |
|---|---|---|---|---|---|---|
| comp01 | 377* | 272 | 151 | 125 | **17** | 21 |
| comp02 | 671* | 657 | 644 | 635 | **221** | 250 |
| comp03 | 715* | 610 | 598 | 567 | 233 | **217** |
| comp04 | 582* | 579 | 559 | 464 | **102** | 155 |
| comp05 | 2011* | 1736* | 1704* | 1614* | 872* | 641* |
| comp06 | 789* | 701* | 765* | 777* | **174** | 262 |
| comp07 | 837* | 794 | 794 | 766 | **180** | 309 |
| Средний ранг | 6.00 | 4.64 | 4.07 | 3.29 | **1.29** | 1.71 |

## Структура

```
code/
  ctt.py             модель задачи, чтение экземпляров, полная и инкрементальная оценка
  algorithms.py      Greedy-Sat, Greedy-Rand, HC, SA, GA, MA
  run_experiment.py  эксперимент (параллельно, бюджет по процессорному времени)
  analyze.py         таблицы, тесты Фридмана и Уилкоксона, графики
  smoke_test.py      проверка: инкрементальная оценка == полный пересчёт
results/             runs.csv (все запуски), curves.json (сходимость), summary.json, графики
```

## Воспроизведение

```bash
pip install numpy pandas scipy matplotlib
# экземпляры ITC-2007 из коллекции Potassco (лицензия MIT)
curl -LO https://stbenchmarkprosc01.blob.core.windows.net/public/cbctt.tar.gz
mkdir -p data/raw && tar -xzf cbctt.tar.gz -C data/raw
python code/smoke_test.py data/raw/CurriculumBasedCourseTimetabling/Instances/ITC-2007_asp/comp01.lp
python code/run_experiment.py data/raw/CurriculumBasedCourseTimetabling/Instances/ITC-2007_asp results 60 3 3
cd code && python analyze.py ../results ../paper
```
