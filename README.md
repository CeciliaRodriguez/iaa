# Inteligencia Artificial Aplicada

## Build docker image
```bash
docker build -t iaa .
```
## Run in docker 
```bash
docker run -it -v ${PWD}:/var/app -w /var/app iaa bash
```
## No docker 
```bash
python -m pip install -U -r requirements.txt
```
## Entrenar y evaluar red
```bash
python entrenar_evaluar.py
```
## Entrenar el modelo con todo el dataset y guardarlo
```bash
python entrenar_guardar.py
```
## Predecimos sobre las imagenes en el directorio test
```bash
python predecir.py
```
