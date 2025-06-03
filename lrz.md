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

## FoldToken installation
Hier ein bewährter Weg um FoldToken auf dem lrz zum Laufen zu bringen!
1. Git klonen
   -  Wir klonen nicht das Original, sondern ein Fork mit einer funktionierenden Installation
   ```bash
   git clone https://github.com/mahdip72/FoldToken_open.git 
    ```
   - hier interessiert uns eigentlich nur `foldtoken/installation.sh`
2. Enroot Container erstellen
   - Wir benötigen einen Container mit CUDA 11.7
   - Am besten wäre es, wenn dieser auch Python 3.9.17 hätte, wir "umgehen" hier das Ganze mit conda
   - Hier ein Container mit dem es klappt:
   ```bash
   enroot import docker://nvcr.io/nvidia/pytorch:22.05-py3
   ```
   ```bash
   enroot create --name foldtoken nvidia+pytorch+22.05-py3.sqsh
   ```
   ```bash
   enroot start foldtoken
   ```
3. Ressourcen anfragen
   - Hier empfehle ich eine A100 GPU zu nehmen, da `FlashAttention` diese braucht
   ```bash
   salloc -p lrz-dgx-a100-80x8 --gres=gpu:1
   ```
   ```bash
   srun --pty bash
   ```
4. Environment erstellen
  - Erst Environment mit python 3.9.17 erstellen und starten
  ```bash
  conda create -n  f39 python=3.9.17
  ```
  ```bash
  conda activate f39
  ```
  - Jetzt navigiere ins vorher geklonte Repo in den FoldToken Ordner und lasse die `installation.sh` laufen
  ```bash
  bash installation.sh
  ```
  - Hier meckert vermutlich `flash-attention`, um das zu lösen, installiere es mit den folgenden Befehlen per wheel und dann lass die Installation noch mal laufen
  ```bash
  wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
  pip install ./flash_attn-2.6.3+cu118torch2.0cxx11abiFALSE-cp39-cp39-linux_x86_64.whl --no-build-isolation
  ```
  - Sobald `installation.sh` durchgelaufen ist, sind alle packages installiert

5. FoldToken ausführen
  - Jetzt muss noch das auf GitHub verlinkte Model heruntergeladen und entpackt werden
  - Und in der `config.yaml` im Ordner von Model die Zeile `k_neighbors: 30` eingefügt werden 
  - Wenn man jetzt FoldToken versucht laufen zu lassen wie unten, kann es zu einem `Segmentation Fault` kommen. Hier einfach:
  ```bash
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
  ```
  - Mit diesen Schritten kann das original Repro als auch das editierte ausgeführt werden :) 
  ```
CUDA_VISIBLE_DEVICES=0 python foldtoken/reconstruct.py --path_in casp14/ --path_out casp14_out --level 10
  ```

## Sonstige nützliche Commands

- `du -h --max-depth=2` listed directoriers auf und wie viel Speicherplatz sie verbrauchen
- `nvcc --version` nvidia cuda version


