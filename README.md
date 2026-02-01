# Music-Clasterizer
Final Project for Machine Learning 
## Environment setup

```bash
conda env create -f environment.yml
conda activate audio_env
python -m ipykernel install --user --name audio_env --display-name "Python (audio_env)"
```

## Environemnt update
Dodawanie bibliotek do naszego środowiska odbywa się poprzez dodawanie ich po myślniku do pliku environment.yml
Następnie aby zaktualizować to środowisko u siebie lokalnie, wystarczy użyć komendy w terminalu:
```bash
conda env update -n audio_env --file environment.yml
```

## Dodawanie piosenek do datasetu
Wrzucamy w folder 'data' jakieś pliki .mp3 i odpalamy funkcję ```preprocess_dataset``` w notebooku representations.ipynb