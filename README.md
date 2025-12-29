# Modul Data Wrangling: Energie & Wetter 

Dieses Repository enthält die Ergebnisse unserer Gruppenarbeit im Modul "Data Wrangling". Ziel des Projekts war die Entwicklung einer robusten Daten-Pipeline, die Energieverbrauchsdaten und Wetterdaten bereinigt, transformiert und mittels verschiedener statistischer Methoden auf eine Basis aggregiert.

## Schnellstart 

Die zentrale Datei zur Ausführung der gesamten Pipeline und zur Reproduktion der Ergebnisse ist:

👉 **`LE5_Pipeline_Comparison.ipynb`**

Dieses Notebook führt alle Schritte – vom Download der Rohdaten über die Bereinigung bis hin zur finalen Aggregation und Speicherung – automatisiert aus.

---

## Projektstruktur

Das Projekt ist in Lerneinheiten (LE) unterteilt, die den Entwicklungsprozess dokumentieren:

* **`LE5_Pipeline_Comparison.ipynb`** (**Main File**): Die finale, modulare Pipeline. Beinhaltet die Funktionen zur Rohdatenakquise, Bereinigung und Anwendung der Aggregationsmethoden.
* `requirements.txt`: Liste der benötigten Python-Bibliotheken.
* `data`-file: Daten und Outputs sind hier abgespeichert
* `notebooks`-file: Alle Python notebooks (auch das LE5_Pipeline_Comparison.ipynb

## Installation & Setup

Um den Code auszuführen, folgen Sie diesen Schritten:

1.  **Repository klonen:**
    ```bash
    git clone https://github.com/r9119/DAW_MC.git
    cd DAW_MC
    ```

2.  **Abhängigkeiten installieren:**
    Es wird empfohlen, eine virtuelle Umgebung zu nutzen. Installieren Sie danach die Requirements:
    ```bash
    pip install -r requirements.txt
    ```

## Ausführung & Reproduzierbarkeit

1.  Öffnen Sie das Notebook **`LE5_Pipeline_Comparison.ipynb`** in Jupyter Notebook, JupyterLab oder PyCharm.
2.  Führen Sie alle Zellen aus ("Run All").
3.  Die Pipeline führt folgende Schritte automatisch durch:
    * Download des Datasets via `kagglehub` (Internetverbindung erforderlich).
    * Bereinigung (Duplikate entfernen, Missing Values imputieren).
    * Aggregation der Zeitreihen auf Tagesbasis.

---

## Autoren

* Rami Tarabishi
* Pascal Trösch
* Ilyas Kayihan
