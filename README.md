# audio2text

Превращает аудио- или видеофайл либо ссылку на ролик YouTube в текст в формате Markdown: кто говорил
(**SPEAKER_00**, **SPEAKER_01**, ...), когда и что. Всё считается на вашем компьютере, без облака и без видеокарты.
Работает на macOS, Windows и Linux. Хватает 8 ГБ памяти.

## Установка

Откройте терминал (macOS: программа «Терминал»; Windows: «PowerShell»; Linux: любой терминал), вставьте одну
команду и нажмите Enter.

macOS и Linux:

```
curl -LsSf https://astral.sh/uv/install.sh | sh && ~/.local/bin/uv tool install https://github.com/tumikosha/audio2text/archive/refs/heads/main.zip
```

Windows (PowerShell):

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"; & "$env:USERPROFILE\.local\bin\uv.exe" tool install https://github.com/tumikosha/audio2text/archive/refs/heads/main.zip
```

Команда ставит uv (он сам скачает Python и всё остальное), а затем audio2text. git не нужен. Если uv уже
установлен, команда его просто обновит.

Затем закройте терминал и откройте его снова. Проверьте: `audio2text --help` должен показать справку. Если
терминал пишет, что команда не найдена, выполните `~/.local/bin/uv tool update-shell` (Windows:
`& "$env:USERPROFILE\.local\bin\uv.exe" tool update-shell`), закройте и снова откройте терминал.

## Как пользоваться

Общий вид команды:

```
audio2text <файл или ссылка> [итог.md] [-tr] [--video] [--mp3] [--model M] [--keep]
```

| Что указать | Зачем |
|---|---|
| `<файл или ссылка>` | обязательно: аудио- или видеофайл с диска либо ссылка на один ролик YouTube |
| `итог.md` | куда сохранить текст; без него — `<имя файла>.md` или `<название ролика>.md` в текущей папке |
| `-tr` (или `--transcribe`) | для ссылки: не брать субтитры, а распознать звук с разделением на говорящих |
| `--video` | для ссылки: сохранить ещё и сам ролик, `итог.mp4` рядом с текстом |
| `--mp3` | сохранить ещё и звук, `итог.mp3` рядом с текстом (для ссылки и для файла) |
| `--model M` | другая модель распознавания: `parakeet-tdt-0.6b-v2` или папка со своей моделью, см. [Другая модель](#другая-модель) |
| `--keep` | не удалять временные файлы (скачанный звук) после успешной работы |
| `--help` | показать справку |

Примеры:

```
audio2text запись.mp3
audio2text интервью.mp4 итог.md
audio2text "https://www.youtube.com/watch?v=jNQXAC9IVRw"
audio2text "https://www.youtube.com/watch?v=jNQXAC9IVRw" -tr
audio2text "https://www.youtube.com/watch?v=jNQXAC9IVRw" --video --mp3
audio2text english_talk.m4a --model parakeet-tdt-0.6b-v2
```

Так выглядит итоговый файл после распознавания:

```markdown
# запись

- Источник: запись.mp3
- Транскрипт: локально, parakeet-tdt-0.6b-v3
- Длительность: 0:12:41

**SPEAKER_00** `00:00:00` Добрый день. Спасибо, что нашли время.

**SPEAKER_01** `00:00:04` Здравствуйте, рад помочь.
```

Новый абзац начинается при смене говорящего, а длинная речь одного человека делится на абзацы примерно
по минуте. Из субтитров YouTube текст получается без имён говорящих.

Подробнее о каждом случае ниже.

Файл с диска (mp3, m4a, wav, mp4 и другие форматы, которые понимает ffmpeg):

```
audio2text запись.mp3
```

Текст появится в файле `запись.md` в той папке, где открыт терминал. Можно сразу указать, куда сохранить:

```
audio2text запись.mp3 итог.md
```

Ссылка на ролик YouTube:

```
audio2text "https://www.youtube.com/watch?v=jNQXAC9IVRw"
```

По ссылке программа сначала ищет субтитры на языке ролика. Если они есть, текст получится быстро, но без
разделения на говорящих. Чтобы скачать звук и распознать его самой программой, с говорящими, добавьте `-tr`:

```
audio2text "https://www.youtube.com/watch?v=jNQXAC9IVRw" -tr
```

Ссылку берите в кавычки. Плейлисты и каналы не принимаются, только одно видео.

Чтобы сохранить сам ролик и его звук, добавьте `--video` и (или) `--mp3`:

```
audio2text "https://www.youtube.com/watch?v=jNQXAC9IVRw" --video --mp3
```

Файлы ложатся рядом с текстом и называются так же: `Me at the zoo.md`, `Me at the zoo.mp4`,
`Me at the zoo.mp3`. Видео сохраняется в mp4, который открывают обычные плееры, звук — в mp3. `--mp3` работает
и для файла с диска, `--video` — только для ссылки. Даже если текст берётся из субтитров, ролик или звук всё равно
скачиваются. Уже существующие `.mp4` и `.mp3` программа не перезаписывает: она сообщает об этом и ничего
не делает (для ссылки — ещё до загрузки ролика), удалите старый файл или укажите другое имя итога.

На Windows ссылки YouTube не проверены: автоматическую проверку (CI) YouTube блокирует как бота. Если не работает,
включите запасной путь `AUDIO2TEXT_YTDLP_NO_DENO=1` (нужен установленный deno или Node.js), см.
[Если ссылки YouTube не работают](#если-ссылки-youtube-не-работают).

Во время работы программа пишет длительность записи и примерное время до конца. Если что-то не получилось,
она пишет одну строку с причиной по-русски. Исключение — ошибки в самих параметрах команды (неизвестный ключ,
нет имени файла): подсказку о правильном вызове (`usage: ...`) и строку ошибки выводит стандартный разбор
аргументов Python, на английском.

## Другая модель

По умолчанию речь распознаёт многоязычная модель parakeet-tdt-0.6b-v3. Для записей на английском можно взять
модель parakeet-tdt-0.6b-v2 для английского:

```
audio2text запись.mp3 --model parakeet-tdt-0.6b-v2
```

При первом запуске с ней программа один раз докачает около 0,48 ГБ. Для ссылок YouTube `--model` действует, только
когда текст распознаётся программой (нет субтитров или указан `-tr`).

Можно указать папку со своей моделью: модель NeMo transducer (Parakeet) в формате sherpa-onnx; другие типы
(zipformer и т.п.) не поддерживаются.

```
audio2text запись.mp3 --model путь/к/папке
```

Если в пути есть пробелы, возьмите его в кавычки: `--model "C:\Мои модели\parakeet"`.

В папке должны быть файлы `encoder.onnx`, `decoder.onnx`, `joiner.onnx` и `tokens.txt`. Вместо каждого из трёх
файлов `.onnx` может лежать его вариант `.int8.onnx`, он берётся в первую очередь. Ничего не скачивается.
В строке «Транскрипт» итогового файла будет имя модели или имя папки.

## Первый запуск

При первом запуске программа один раз скачает модели распознавания: около 0,53 ГБ (на диске займут
около 0,7 ГБ). Нужен интернет. Если загрузка оборвалась, запустите ту же команду ещё раз: загрузка продолжится
с того же места. Потом модели берутся с диска, интернет для файлов уже не нужен (для ссылок YouTube нужен всегда).

## Где лежат модели

Модели и временные файлы лежат в папке кэша:

| Система | Папка |
|---|---|
| macOS | `~/Library/Caches/audio2text` |
| Linux | `~/.cache/audio2text` |
| Windows | `%LOCALAPPDATA%\audio2text\audio2text\Cache` |

Внутри: `models` — модели, `work` — временные файлы (скачанный звук роликов; после успешной работы
удаляются сами). Папку можно удалить целиком, это безопасно: при следующем запуске модели скачаются заново.
Другую папку кэша можно задать переменной окружения `AUDIO2TEXT_CACHE`.

## Удаление

Одна команда удаляет программу вместе с моделями и временными файлами.

macOS:

```
rm -rf ~/Library/Caches/audio2text && ~/.local/bin/uv tool uninstall audio2text
```

Linux:

```
rm -rf "${XDG_CACHE_HOME:-$HOME/.cache}/audio2text" && ~/.local/bin/uv tool uninstall audio2text
```

Windows (PowerShell):

```
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\audio2text" -ErrorAction SilentlyContinue; & "$env:USERPROFILE\.local\bin\uv.exe" tool uninstall audio2text
```

Если вы задавали свою папку кэша через `AUDIO2TEXT_CACHE`, удалите и её. Сам uv остаётся, он может быть нужен
другим программам.

## Обновление

Та же команда установки с `--reinstall` скачивает свежую версию и ставит её поверх старой:

```
uv tool install --reinstall https://github.com/tumikosha/audio2text/archive/refs/heads/main.zip
```

Модели при этом заново не скачиваются.

## Если ссылки YouTube не работают

Для YouTube программа запускает yt-dlp вместе с JS-движком deno, всё скачивается само. Если на вашей системе
это не работает, установите deno (https://deno.com) или Node.js (https://nodejs.org) и включите запасной путь.

macOS и Linux:

```
AUDIO2TEXT_YTDLP_NO_DENO=1 audio2text "https://www.youtube.com/watch?v=jNQXAC9IVRw"
```

Windows (PowerShell):

```
$env:AUDIO2TEXT_YTDLP_NO_DENO = "1"
audio2text "https://www.youtube.com/watch?v=jNQXAC9IVRw"
```

## Длинные записи: память и время

Замер на 3-часовой записи (mp3, склейка коротких публичных фрагментов речи нескольких людей с паузами),
команда `/usr/bin/time -l audio2text long.mp3 out.md`, macOS, Apple M4 Max, 14 ядер:

| Версия | Пиковая память (maximum resident set size) | Время |
|---|---|---|
| до исправления чтения аудио | 3,80 ГБ | 9 мин 27 с |
| после исправления (текущая) | 2,94 ГБ | 9 мин 02 с |

Аудио 3 часа в памяти — около 0,7 ГБ; раньше при чтении держались ещё две его копии. Предел для 8 ГБ памяти —
4 ГБ, запись целиком укладывается, резать её на части не нужно. Результат обеих версий совпал.
На компьютере с 2–4 ядрами распознавание идёт дольше; на таком железе время не замерялось. Перед работой программа
сама пишет оценку времени для вашей записи.

## Модели

| Задача | Модель | Лицензия |
|---|---|---|
| Распознавание речи | NVIDIA Parakeet TDT 0.6B v3, int8 (экспорт k2-fsa/sherpa-onnx) | CC-BY-4.0 |
| Распознавание речи, английский (`--model parakeet-tdt-0.6b-v2`) | NVIDIA Parakeet TDT 0.6B v2, int8 (экспорт k2-fsa/sherpa-onnx) | CC-BY-4.0 |
| Поиск речи | Silero VAD | MIT |
| Говорящие | NVIDIA NeMo TitaNet small (экспорт k2-fsa/sherpa-onnx), эмбеддинг 192 | Apache-2.0 (лицензия NeMo Toolkit, карточка NGC nvidia/nemo/titanet_small) |

Модели скачиваются один раз в кэш пользователя и проверяются по sha256.

## Как выбрана модель спикеров

Голос описывается эмбеддингом окна 1.5 с (шаг 0.75 с). Окна группируются косинусным k-means.
Число говорящих от 1 до 4 выбирается по среднему косинусному силуэту. Если лучший силуэт ниже `SIL_MIN` = 0.1,
говорящий один. Кандидаты: ONNX-модели эмбеддингов голоса из релиза sherpa-onnx.
Сравнение шло на одних и тех же сегментах распознавания, менялась только модель эмбеддингов.
За точку отсчёта взята модель SpeechBrain ECAPA-TDNN (torch).

Метрика: доля размеченной речи, отнесённой к правильному говорящему. Каждый найденный говорящий сопоставлен
той стороне разговора, с которой он больше всего пересекается. Набор: 145 размеченных телефонных разговоров,
около 7 400 с речи.

| Модель | `SIL_MIN` 0.05 | 0.1 | 0.15 | 0.2 |
|---|---|---|---|---|
| ECAPA-TDNN (точка отсчёта, torch) | 90.22% | 90.22% | 89.30% | 83.30% |
| **NeMo TitaNet small** | 90.93% | **90.93%** | 90.93% | 89.80% |
| 3D-Speaker CAM++ (VoxCeleb) | 90.09% | 90.09% | 90.09% | 88.45% |
| 3D-Speaker ERes2Net base (zh-cn) | 89.97% | 89.97% | 87.35% | 81.05% |
| WeSpeaker ResNet34 (VoxCeleb) | 84.41% | 84.41% | 84.41% | 82.48% |

Правило выбора: лучшая ONNX-модель, если она не хуже ECAPA больше чем на 1 п. п. Выбрана TitaNet small, `SIL_MIN` 0.1.

Ограничения оценки:

- Набор — только телефонные разговоры двух людей. На записях с одним говорящим, с тремя и больше,
  на музыке или студийном звуке модель и порог `SIL_MIN` не проверялись.
- Метрика не штрафует за лишнее дробление: если один человек разбит на двух говорящих, это не считается ошибкой.
  Поэтому порог 0.1 на этом наборе не показывает, насколько надёжно распознаётся запись с одним голосом.

## Атрибуция моделей

- Распознавание речи: NVIDIA parakeet-tdt-0.6b-v3 (https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), int8-экспорт
  проекта sherpa-onnx. Текст атрибуции из README пакета модели:

  > The weights are NVIDIA's parakeet-tdt-0.6b-v3, trained by NVIDIA and licensed under
  > [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This ONNX export was produced by the sherpa-onnx
  > project, whose code is licensed under [Apache-2.0](https://github.com/k2-fsa/sherpa-onnx/blob/master/LICENSE).
  > NVIDIA does not endorse this package.

- Распознавание речи на английском (`--model parakeet-tdt-0.6b-v2`): NVIDIA parakeet-tdt-0.6b-v2
  (https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), int8-экспорт проекта sherpa-onnx. Текст атрибуции
  из README пакета модели:

  > The weights are NVIDIA's parakeet-tdt-0.6b-v2, trained by NVIDIA and licensed under
  > [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This ONNX export was produced by the sherpa-onnx
  > project, whose code is licensed under [Apache-2.0](https://github.com/k2-fsa/sherpa-onnx/blob/master/LICENSE).
  > NVIDIA does not endorse this package.

- Поиск речи: Silero VAD (https://github.com/snakers4/silero-vad), лицензия MIT.
- Говорящие: NVIDIA NeMo TitaNet small (https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_small),
  ONNX-экспорт проекта sherpa-onnx, лицензия Apache-2.0 (лицензия NeMo Toolkit).

Модели скачиваются из релизов https://github.com/k2-fsa/sherpa-onnx/releases и в репозиторий не входят.

## Тестовый файл

Тесты и проверка в CI берут английскую запись (один говорящий, около 4 с) — файл `test_wavs/en.wav` из пакета
модели sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8:
https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/main/test_wavs/en.wav

Файл скачивается при запуске (тесты кладут его в кэш pytest) и в репозиторий не входит: отдельные условия для
`test_wavs` в пакете не указаны. Карточка репозитория на Hugging Face объявляет для всего пакета CC-BY-4.0;
README пакета говорит только о весах модели и коде экспорта, про записи в `test_wavs` в нём ничего нет.

## Для разработчика

```
uv run pytest -q
```

Тесты с пометкой `models` скачивают настоящие модели (один раз). CI (`.github/workflows/ci.yml`) гоняет тесты
и установку `uv tool install .` на macOS, Windows и Linux.

Лицензия кода не указана.
