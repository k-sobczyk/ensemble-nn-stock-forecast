 
# Krok 1: Analiza dokumnetów.


# Krok 2: Tworzenie DataFrame i uzupełnienie danych
# Czyszczenie danych

WSZYSTKIE PLIKI SĄ IDENTYCZNE. WSZYSTKIE TO EXCEL - 14 kart (my potrzebujemy jedynie 'Info' i 'QS')

Wyciągnięcie z każdego pliku: 'Info'

```python
def test_process_excel_files(folder_path):
    results = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(folder_path, filename)
            try:
                with pd.ExcelFile(file_path) as xls:
                    company_name = pd.read_excel(xls, 'Info', usecols="B", skiprows=1, nrows=1).values[0][0]
                    sector = pd.read_excel(xls, 'Info', usecols="E", skiprows=19, nrows=1).values[0][0]
                
                result = {
                    'filename': filename,
                    'Company Name': company_name,
                    'Sector': sector
                }
                results.append(result)
                
                print(f"Dane z {filename} zostały przetworzone.")
            except Exception as e:
                print(f"Błąd przy przetwarzaniu pliku {filename}: {e}")

    results_df = pd.DataFrame(results)
    return results_df
```
Output:


* czasami w kolumnie z 'Sektor' wystepuje Nan ale jest go tak mało ze równie dobrze mozemy go wypełnić ręcznie


# TODO zrobienie nowej funckji która wyciaga reszte danych a potem łaczy z pierwszym df'em np. po nazwie pliku

Podsumowanie co wyciągnąć z pliku:
ZAKŁADKA Info (done)
    Nazwa
    Sektor


ZAKŁADKA QS (do zrobienia)
    Dodatkowa kolumna jako nazwa pliku
    C – nazwa kolumny w DF

    Wiersze: Od 3 do 18 włącznie
    wiersze 30 do 93 włącznie
    Wiersze 255 – 275 włącznie
    Wiersze 279 – 287 włącznie
    Wiersze 290 – 293 włącznie 

# Krok 3: Feature Enginnering
https://www.gpw.pl/archiwum-notowan?fetch=0&type=10&instrument=&date=10-01-2024&show_x=Poka%C5%BC+wyniki - potencjalnie źródło pozyskania wartości spółki w konkretnym dniu

BONUS:
Czyszczenie:
•	Wszystkie NaN na 0
•	Skalowanie przed modelem
•	Inżynieria cech 
TESTY DO PRZEPROWADZENIA:
•	Uwzglednienie wszystkiego
•	Skupienie sie na zmianach kapitałowych
•	Ograniczenie tylko główne wartości
•	Grupowanie po przedsiębiorstwie? 

