# lrz guide
Hier ein kleiner Spicker zum lrz :)

## Einfacher Workflow 
### 1. Ressourcen requesten
- Einfach nach Ressourcen fragen
- Dabei kannst zwischen GPUS wählen, hier findest du die [Liste](https://doku.lrz.de/1-general-description-and-resources-10746641.html)
- Hier requesten wir lrz-v100x2
```bash
salloc -p lrz-v100x2 --gres=gpu:1
```
- Sobald die Allocation gegranted wird, musst die sie noch "starten":
```bash
srun --pty bash
```
- wenn alles geklappt hat steht jetz "@gpu" bei deinem user stehen:
```bash
ge43vab2@gpu-005:~$
```

### 2. Import Container
- Suche dir irgendeinen Container aus dem Nvidia Katalog oder auch irgendwo anders aus
- Importiere ihn aufs lrz mit
```bash
enroot import docker://<container url>
```
- Das erzeugt einen .sqsh file, den wir jetzt benutzen können
-  Zum Beispiel:
```bash
enroot import docker://nvcr.io/nvidia/pytorch:24.12-py3
```
 
### 3. Create enroot Container

- Einfach einen enroot Container erzeugen mit dem .sqsh file, den wir runtergeladen haben
```bash
enroot create --name my_container nvidia+pytorch+24.12-py3.sqsh
```
- Der kann jetzt gestartet werden
```bash
enroot start my_container
```
- Mögliche Argumente
    - `--root` würde root ermöglichen
    - `--mount <pfad>` mountet eine andere plate, nötig, wenn datein nicht im Home-Verzeichniss liegen

### 4. Export Container
- Um den Container zu speichern, verlasse hin erst mit
```bash
exit
```
- Hier sollte immer noch @gpu stehen!!
- Jetzt exportieren mit
```bash
enroot export --output my_container.sqsh my_container
```
- Das erzeugt einen .sqsh File, den man mit `create` und `start` wieder benutzen kann 

## Sonstige nützliche Commands

- `du -h --max-depth=2` listed directoriers auf und wie viel Speicherplatz sie verbrauchen
- `nvcc --version` nvidia cuda version


