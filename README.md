# MNIST Fashion Classification

Progetto di classificazione del dataset Fashion-MNIST utilizzando PyTorch Lightning e una rete neurale convoluzionale.

## Descrizione

Il repository implementa una pipeline di addestramento completa che include:

- caricamento e gestione del dataset tramite Hugging Face
- definizione di un dataset PyTorch personalizzato
- implementazione di una CNN in PyTorch Lightning
- training e validazione con metriche di performance

## Struttura del progetto

La struttura consigliata del progetto è la seguente:

├── src/
│ ├── dataset.py
│ ├── model.py
│ └── train.py
├── requirements.txt
└── README.md

- `dataset.py`: definizione del dataset e dataloader
- `model.py`: implementazione del modello CNN e training loop Lightning
- `train.py`: script di esecuzione del training

## Requisiti

Creare un ambiente Python ed installare le dipendenze:

pip install -r requirements.txt

## Utilizzo

Avvio del training:
python src/train.py

Il training produce loss e accuracy per training e validazione.

Risultati attesi

Il modello raggiunge tipicamente:

training accuracy intorno al 90%

validation accuracy intorno al 92–93%

Le prestazioni dipendono da parametri, batch size e configurazione hardware.