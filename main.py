"""
1.2. Dodanie wszystkich niezbędnych bibliotek wymaganych do prawidłowego uruchomienia aplikacji
"""
import tkinter as tk  # GUI: główna biblioteka do tworzenia okienek
from tkinter import filedialog, messagebox, simpledialog, ttk  # GUI: dialogi plików, komunikaty, proste dialogi, widżety rozszerzone
from PIL import Image, ImageTk  # Przetwarzanie obrazów: ładowanie i wyświetlanie obrazów w Tkinter
import cv2  # OpenCV: zaawansowane przetwarzanie obrazów i wideo
import numpy as np  # NumPy: operacje na tablicach i macierzach, obliczenia numeryczne
import matplotlib.pyplot as plt  # Matplotlib: tworzenie wykresów i wizualizacji danych
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Integracja wykresów Matplotlib z Tkinter
import os  # Operacje na plikach i ścieżkach systemowych
from skimage.draw import line  # Skimage: rysowanie linii na obrazach
import pickle  # Serializacja: zapisywanie i wczytywanie obiektów Pythona do pliku
import csv  # Obsługa plików CSV: czytanie i zapisywanie danych tabelarycznych
import io  # Operacje na strumieniach wejścia/wyjścia (np. pliki w pamięci)
import sys  # Dostęp do funkcji systemowych i argumentów linii poleceń
import subprocess  # Uruchamianie zewnętrznych procesów/systemowych poleceń
import tempfile  # Tworzenie tymczasowych plików i katalogów

class ImageEditorApp:
    """
    Główna klasa aplikacji do przetwarzania obrazów.
    Zawiera metody do wczytywania, przetwarzania, analizy i wizualizacji obrazów.
    """
    def __init__(self, root):
        """
        Inicjalizacja aplikacji, ustawienie GUI i zmiennych stanu.
        """
        self.root = root
        self.root.title("Aplikacja do przetwarzania obrazów | Oleh Buriakivskyi")
        self.root.minsize(1200, 800)
        self.image = None
        self.gray_image = None
        self.original_image = None
        self.original_gray = None
        self.zoom_factor = 1.0
        self.hist_canvas = None
        self.history = []
        self.profile_points = []
        self.profile_line = None
        self.profile_start = None
        self.profile_end = None
        self.profile_moving = False
        self.move_start = None
        self.image_position = (0, 0)
        self.modified = False
        self.crop_points = []
        self.cropping = False

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.canvas_bindings()

    def create_widgets(self):
        # Tworzy menu, panele i przyciski aplikacji. Każda sekcja menu jest opisana.
        toolbar = tk.Frame(self.root)
        toolbar.pack(side="top", fill="x")
        undo_btn = tk.Button(toolbar, text="⮌", command=self.undo_image)
        undo_btn.pack(side="right", padx=2)

        menubar = tk.Menu(self.root)
        # Menu Plik: wczytywanie, zapisywanie, zamykanie
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Nowe okno", command=self.open_new_window)
        file_menu.add_command(label="Duplikuj", command=self.duplicate_window)
        file_menu.add_separator()
        file_menu.add_command(label="Wczytaj obraz RGB", command=self.load_rgb_image)
        file_menu.add_command(label="Wczytaj obraz szaroodcieniowy", command=self.load_gray_image)
        file_menu.add_separator()
        file_menu.add_command(label="Zapisz obraz", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Zamknij", command=self.on_closing)
        menubar.add_cascade(label="Plik", menu=file_menu)

        # Dodaję osobną pozycję 'O autorze' przed Lab1
        menubar.add_command(label="Info o programie", command=self.show_about_program)

        # Menu Lab1: operacje histogramowe, konwersje, negacja, posteryzacja
        lab1_menu = tk.Menu(menubar, tearoff=0)
        lab1_menu.add_command(label="Na szaro (Gray)", command=self.convert_to_gray)
        lab1_menu.add_command(label="Kanały RGB", command=self.split_rgb_channels)
        lab1_menu.add_command(label="Konwersja RGB → HSV lub Lab", command=self.convert_to_hsv_or_lab)
        lab1_menu.add_separator()
        lab1_menu.add_command(label="Rozciąganie histogramu", command=self.stretch_histogram_manual)
        lab1_menu.add_command(label="Equalizacja histogramu", command=self.equalize_histogram_manual)
        lab1_menu.add_separator()
        lab1_menu.add_command(label="Negacja", command=self.negative_image)
        lab1_menu.add_command(label="Rozciąganie zakresu", command=self.stretch_manual_dialog)
        menubar.add_cascade(label="Lab1", menu=lab1_menu)

        # Menu Lab2: operacje punktowe, sąsiedztwa, maski, medianowa, dwuargumentowe
        lab2_menu = tk.Menu(menubar, tearoff=0)
        posterize_menu = tk.Menu(lab2_menu, tearoff=0)
        for levels in [2, 4, 8, 16, 32, 64, 128]:
            posterize_menu.add_command(label=f"{levels} poziomów", command=lambda l=levels: self.posterize(l))
        lab2_menu.add_cascade(label="Posteryzacja", menu=posterize_menu)
        lab2_menu.add_separator()
        blur_menu = tk.Menu(lab2_menu, tearoff=0)
        blur_menu.add_command(label="Średnia (Blur)", command=lambda: self.apply_blur('blur'))
        blur_menu.add_command(label="Gaussowskie", command=lambda: self.apply_blur('gauss'))
        lab2_menu.add_cascade(label="Wygładzanie", menu=blur_menu)
        edge_menu = tk.Menu(lab2_menu, tearoff=0)
        edge_menu.add_command(label="Sobel", command=lambda: self.apply_edge_detection('sobel'))
        edge_menu.add_command(label="Laplacian", command=lambda: self.apply_edge_detection('laplacian'))
        edge_menu.add_command(label="Canny", command=lambda: self.apply_edge_detection('canny'))
        lab2_menu.add_cascade(label="Detekcja krawędzi", menu=edge_menu)
        sharpen_menu = tk.Menu(lab2_menu, tearoff=0)
        sharpen_menu.add_command(label="Maska 1", command=lambda: self.apply_sharpen(1))
        sharpen_menu.add_command(label="Maska 2", command=lambda: self.apply_sharpen(2))
        sharpen_menu.add_command(label="Maska 3", command=lambda: self.apply_sharpen(3))
        lab2_menu.add_cascade(label="Wyostrzanie", menu=sharpen_menu)
        prewitt_menu = tk.Menu(lab2_menu, tearoff=0)
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        for i, angle in enumerate(angles, 1):
            prewitt_menu.add_command(label=f"Kierunek {i} ({angle}°)", command=lambda d=i: self.apply_prewitt(d))
        lab2_menu.add_cascade(label="Detekcja Prewitta", menu=prewitt_menu)
        lab2_menu.add_command(label="Uniwersalna maska 3x3", command=self.custom_mask_dialog)
        lab2_menu.add_separator()
        lab2_menu.add_command(label="Filtracja medianowa", command=self.median_filter_dialog)
        lab2_menu.add_separator()
        binary_menu = tk.Menu(lab2_menu, tearoff=0)
        binary_menu.add_command(label="Dodawanie", command=self.binary_add)
        binary_menu.add_command(label="Odejmowanie", command=self.binary_subtract)
        binary_menu.add_command(label="Mieszanie (Blending)", command=self.binary_blend_dialog)
        binary_menu.add_command(label="AND", command=self.binary_and)
        binary_menu.add_command(label="OR", command=self.binary_or)
        binary_menu.add_command(label="XOR", command=self.binary_xor)
        lab2_menu.add_cascade(label="Operacje dwuargumentowe", menu=binary_menu)
        lab2_menu.add_separator()
        lab2_menu.add_command(label="Filtracja dwuetapowa i jednoetapowa", command=self.compare_two_step_and_one_step_filter)
        menubar.add_cascade(label="Lab2", menu=lab2_menu)

        # Menu Lab3: morfologia, szkielet, Hough, linia profilu, piramida
        lab3_menu = tk.Menu(menubar, tearoff=0)
        morph_menu = tk.Menu(lab3_menu, tearoff=0)
        morph_menu.add_command(label="Erozja", command=lambda: self.morphological_operation("Erozja"))
        morph_menu.add_command(label="Dylatacja", command=lambda: self.morphological_operation("Dylatacja"))
        morph_menu.add_command(label="Otwarcie", command=lambda: self.morphological_operation("Otwarcie"))
        morph_menu.add_command(label="Zamknięcie", command=lambda: self.morphological_operation("Zamknięcie"))
        lab3_menu.add_cascade(label="Morfologia", menu=morph_menu)
        lab3_menu.add_separator()
        lab3_menu.add_command(label="Szkieletyzacja (Zhang-Suen)", command=self.thinning_dialog)
        lab3_menu.add_separator()
        hough_menu = tk.Menu(lab3_menu, tearoff=0)
        hough_menu.add_command(label="Detekcja linii", command=self.detect_lines_hough)
        hough_menu.add_command(label="Parametry detekcji", command=self.hough_parameters_dialog)
        lab3_menu.add_cascade(label="Transformata Hougha", menu=hough_menu)
        lab3_menu.add_separator()
        lab3_menu.add_command(label="Linia profilu", command=self.start_profile_selection)
        lab3_menu.add_separator()
        lab3_menu.add_command(label="Piramida obrazów", command=self.image_pyramid)
        menubar.add_cascade(label="Lab3", menu=lab3_menu)

        # Menu Lab4: segmentacja, inpainting, kompresja, analiza cech
        lab4_menu = tk.Menu(menubar, tearoff=0)
        segmentation_menu = tk.Menu(lab4_menu, tearoff=0)
        segmentation_menu.add_command(label="Progowanie ręczne", command=self.manual_threshold)
        segmentation_menu.add_command(label="Progowanie adaptacyjne", command=self.adaptive_threshold)
        segmentation_menu.add_command(label="Progowanie Otsu", command=self.otsu_threshold)
        lab4_menu.add_cascade(label="Segmentacja podstawowa", menu=segmentation_menu)
        lab4_menu.add_separator()
        lab4_menu.add_command(label="Segmentacja GrabCut", command=self.grabcut_segmentation)
        lab4_menu.add_separator()
        lab4_menu.add_command(label="Segmentacja Watershed", command=self.watershed_segmentation)
        lab4_menu.add_separator()
        lab4_menu.add_command(label="Inpainting", command=self.inpainting)
        lab4_menu.add_separator()
        rle_menu = tk.Menu(lab4_menu, tearoff=0)
        rle_menu.add_command(label="Kompresja RLE", command=self.rle_compression)
        rle_menu.add_command(label="Zapisz skompresowany", command=self.save_compressed)
        rle_menu.add_command(label="Wczytaj skompresowany", command=self.load_compressed)
        lab4_menu.add_cascade(label="Kompresja RLE", menu=rle_menu)
        lab4_menu.add_separator()
        analysis_menu = tk.Menu(lab4_menu, tearoff=0)
        analysis_menu.add_command(label="Momenty", command=self.calculate_moments)
        analysis_menu.add_command(label="Pole i obwód", command=self.calculate_area_perimeter)
        analysis_menu.add_command(label="Współczynniki kształtu", command=self.calculate_shape_factors)
        analysis_menu.add_command(label="Wektor cech obiektów", command=self.calculate_feature_vector)
        lab4_menu.add_cascade(label="Analiza obrazu", menu=analysis_menu)
        menubar.add_cascade(label="Lab4", menu=lab4_menu)

        # Dodaję zakładkę Projekt
        project_menu = tk.Menu(menubar, tearoff=0)
        project_menu.add_command(label="Kadrowanie prostokątem", command=self.crop_rectangle_drag)
        project_menu.add_separator()
        project_menu.add_command(label="Obrót o 90 stopni w prawo", command=self.rotate_90_right)
        project_menu.add_command(label="Obrót o 90 stopni w lewo", command=self.rotate_90_left)
        project_menu.add_separator()
        project_menu.add_command(label="Obrót o 180 stopni", command=self.rotate_180)
        menubar.add_cascade(label="Projekt", menu=project_menu)

        self.root.config(menu=menubar)

        # Główne panele aplikacji
        main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_frame.pack(fill="both", expand=True)
        left_frame = ttk.Frame(main_frame)
        right_frame = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        main_frame.add(left_frame, weight=7)
        main_frame.add(right_frame, weight=3)
        self.img_panel_container = tk.Canvas(left_frame)
        self.img_panel_container.pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas_image_id = None
        self.hist_frame = ttk.LabelFrame(right_frame, text="Histogram")
        right_frame.add(self.hist_frame, weight=3)
        self.table_frame = ttk.LabelFrame(right_frame, text="Tablica histogramu")
        right_frame.add(self.table_frame, weight=1)
        self.hist_table_scroll = tk.Scrollbar(self.table_frame)
        self.hist_table_scroll.pack(side="right", fill="y")
        self.hist_table = tk.Text(self.table_frame, height=6, width=40, yscrollcommand=self.hist_table_scroll.set)
        self.hist_table.pack(side="left", fill="both", expand=True)
        self.hist_table_scroll.config(command=self.hist_table.yview)
        self._image_reference = None
        self.root.bind("<Control-MouseWheel>", self.zoom_image)

    def display_image(self):
        """
        2.1. Wyświetla obraz na panelu z uwzględnieniem zoomu i centrowania.
        """
        if self.image is None:
            return

        # Czyszczenie canvasa:
        self.img_panel_container.delete("all")

        # Konwersja z BGR (OpenCV) do RGB (PIL/Tkinter)
        img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        container_w = self.img_panel_container.winfo_width()
        container_h = self.img_panel_container.winfo_height()

        # Ograniczenie maksymalnego zoomu, aby obraz nie był większy niż kontener
        if container_w > 0 and container_h > 0:
            scale = min(container_w / w, container_h / h)
            if scale < self.zoom_factor:
                self.zoom_factor = scale * 0.8
        
        # Przeskalowanie obrazu do aktualnego zoomu
        new_size = (int(w * self.zoom_factor), int(h * self.zoom_factor))
        img_resized = cv2.resize(img_rgb, new_size, interpolation=cv2.INTER_LINEAR)
        img_pil = Image.fromarray(img_resized)
        self._image_reference = ImageTk.PhotoImage(img_pil) # Referencja, aby GC nie usunął obrazu

        # Wycentrowanie obrazu na płótnie
        x = max(0, (container_w - new_size[0]) // 2)
        y = max(0, (container_h - new_size[1]) // 2)
        self.image_position = (x, y)

        # Usunięcie starego obrazu i narysowanie nowego na canvasie
        if self.canvas_image_id:
            self.img_panel_container.delete(self.canvas_image_id)
        
        self.canvas_image_id = self.img_panel_container.create_image(
            x, y, anchor="nw", image=self._image_reference
        )

        # Odświeżenie histogramu, jeśli nie są aktywne inne narzędzia
        if not hasattr(self, 'profile_points') or len(self.profile_points) < 2:
            self.show_histogram()

    def calculate_histogram_manual(self, img):
        """
        2.2. Własny algorytm liczenia histogramu obrazu szaroodcieniowego.
        Iteruje po każdym pikselu i zlicza wystąpienia każdej wartości jasności (0-255).
        """
        hist = np.zeros(256, dtype=int)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                hist[img[y, x]] += 1
        return hist

    def show_histogram(self):
        """
        2.2. Wyświetla histogram obrazu w formie graficznej i tabelarycznej.
        """
        # Usuń poprzedni wykres histogramu, jeśli istnieje
        if self.hist_canvas:
            self.hist_canvas.get_tk_widget().destroy()

        # Jeśli nie ma obrazu, nie rób nic
        if self.image is None:
            return
        
        # Jeśli obraz jest kolorowy, histogram liczony jest dla wersji w skali szarości
        if len(self.image.shape) == 3 and self.image.shape[2] == 3:
            img_for_hist = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            img_for_hist = self.image.copy()

        # Oblicz histogram (własna implementacja)
        hist = self.calculate_histogram_manual(img_for_hist)

        # Utwórz wykres histogramu za pomocą matplotlib
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot(hist, color='black')
        ax.set_xlim([0, 255])

        # Osadź wykres w oknie aplikacji (Tkinter)
        self.hist_canvas = FigureCanvasTkAgg(fig, master=self.hist_frame)
        self.hist_canvas.draw()
        self.hist_canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)
        self.show_histogram_table(hist)

    def show_histogram_table(self, hist):
        """
        2.3. Wyświetla histogram w formie tabelarycznej (poziom: ilość pikseli).
        """
        # Wyczyść poprzednią zawartość tabeli
        self.hist_table.delete("1.0", tk.END)

        # Wstaw każdą wartość histogramu jako osobny wiersz
        for i, v in enumerate(hist):
            self.hist_table.insert(tk.END, f"{i}: {v}\n")

    def push_history(self):
        """
        2.4. Dodaje aktualny stan obrazu do historii (do cofania zmian).
        """
        if self.image is not None:
            gray_copy = self.gray_image.copy() if self.gray_image is not None else None  # kopia obrazu szarości
            self.history.append((self.image.copy(), gray_copy))  # dodanie do historii

    def undo_image(self):
        """
        2.4. Cofnięcie ostatniej operacji na obrazie.
        """
        if len(self.history) > 1:
            self.history.pop()
            self.image, self.gray_image = self.history[-1]
            self.display_image()
            self.show_histogram()

    def zoom_image(self, event):
        """
        Obsługuje zoomowanie obrazu za pomocą kółka myszy.
        """
        if event.delta > 0:
            self.zoom_factor *= 1.1
        else:
            self.zoom_factor /= 1.1
        self.display_image()
        self.show_histogram()

    def open_new_window(self):
        """
        3.1.1. Otwiera nowe okno aplikacji (nowa instancja programu).
        """
        subprocess.Popen([sys.executable, sys.argv[0]])

    def duplicate_window(self):
        """
        3.1.2. Otwiera bieżący obraz w nowym oknie programu (duplikacja przez plik tymczasowy).
        """
        # Sprawdzenie, czy jest obraz do duplikacji
        if self.image is None:
            messagebox.showwarning("Brak obrazu", "Nie ma obrazu do zduplikowania.")
            return

        # Przygotowanie danych do przekazania do nowego okna
        data_to_pass = {
            'image': self.image,
            'gray_image': self.gray_image
        }

        # Zapisanie obrazu do tymczasowego pliku pickle
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            pickle.dump(data_to_pass, tmp)
            temp_path = tmp.name

        # Uruchomienie nowej instancji programu z przekazaniem ścieżki do pliku tymczasowego
        subprocess.Popen([sys.executable, sys.argv[0], temp_path])

    def load_from_pickle(self, path):
        """
        3.1.2. Wczytuje obraz z pliku tymczasowego (pickle) i go usuwa.
        """
        try:
            # Odczytanie danych z pliku pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)

            # Przywrócenie obrazu i obrazu szaroodcieniowego
            self.image = data['image']
            self.gray_image = data['gray_image']

            # Zapisanie oryginalnego stanu obrazu i wyczyścienie historii
            self.original_image = self.image.copy()
            self.original_gray = self.gray_image.copy() if self.gray_image is not None else None
            self.history.clear()
            self.push_history()  # zapisz do historii
            self.display_image()  # odśwież obraz
            self.show_histogram()  # odśwież histogram

        except Exception as e:
            # Obsługa błędów podczas wczytywania
            messagebox.showerror("Błąd wczytywania", f"Nie udało się wczytać zduplikowanego obrazu: {e}")
        finally:
            # Usunięcie pliku tymczasowego po wczytaniu
            if os.path.exists(path):
                os.remove(path)

    def load_rgb_image(self):
        """
        3.1.3. Wczytuje obraz z pliku i przygotowuje do dalszej obróbki.
        """
        # Otwórz okno dialogowe do wyboru pliku graficznego
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.bmp;*.jpg;*.png;*.gif")])
        if not path:
            return
        
        # Wczytaj obraz z pliku (z zachowaniem oryginalnych kanałów)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            messagebox.showerror("Błąd", "Nie udało się wczytać obrazu.")
            return
        
        # Dopasowanie początkowego zoomu do rozmiaru okna
        window_width = self.img_panel_container.winfo_width()
        window_height = self.img_panel_container.winfo_height()
        if window_width > 0 and window_height > 0:
            img_height, img_width = img.shape[:2]
            scale = min(window_width / img_width, window_height / img_height)
            self.zoom_factor = scale * 0.8
        
        # Jeśli obraz jest 3-kanałowy, zachowaj go jako BGR.
        # Jeśli ma 2 kanały (lub 1), przekonwertuj go na BGR, aby można było wyświetlać kolorowe linie.
        if len(img.shape) == 3:
            self.image = img
            self.gray_image = None # Obraz nie jest w skali szarości
        else:
            self.gray_image = img
            self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)

        self.original_image = self.image.copy()
        self.original_gray = self.gray_image.copy() if self.gray_image is not None else None
        self.history.clear()
        self.push_history()
        self.display_image()
        self.show_histogram()

    def load_gray_image(self):
        """
        3.1.4. Wczytuje obraz z pliku i przygotowuje do dalszej obróbki.
        """
        # Otwórz okno dialogowe do wyboru pliku graficznego
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.bmp;*.jpg;*.png;*.gif")])
        if not path:
            return
        
        # Wczytaj obraz z pliku (z zachowaniem oryginalnych kanałów)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            messagebox.showerror("Błąd", "Nie udało się wczytać obrazu.")
            return
        
        # Dopasowanie początkowego zoomu do rozmiaru okna
        window_width = self.img_panel_container.winfo_width()
        window_height = self.img_panel_container.winfo_height()
        if window_width > 0 and window_height > 0:
            img_height, img_width = img.shape[:2]
            scale = min(window_width / img_width, window_height / img_height)
            self.zoom_factor = scale * 0.8
        
        # Zawsze konwertuj do skali szarości (nawet jeśli obraz był kolorowy)
        self.gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        # Utwórz wersję 3-kanałową do wyświetlania (potrzebne do rysowania kolorowych linii)
        self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)

        # Zapisz oryginalny stan obrazu i wyczyść historię
        self.original_image = self.image.copy()
        self.original_gray = self.gray_image.copy() if self.gray_image is not None else None
        self.history.clear()
        self.push_history()
        self.display_image()
        self.show_histogram()

    def save_image(self):
        """
        3.1.5. Zapisuje aktualny obraz do pliku (BMP, PNG, JPG).
        """
        if self.image is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".bmp", filetypes=[("BMP", "*.bmp"), ("PNG", "*.png"), ("JPG", "*.jpg")])
        if path:
            self.modified = True  # oznacz jako zmodyfikowany
            cv2.imwrite(path, self.image)  # zapis obrazu do pliku
            messagebox.showinfo("Zapisano", f"Obraz zapisany jako: {path}")  # komunikat o sukcesie

    def on_closing(self):
        """
        3.1.6. Obsługuje zamykanie aplikacji z opcją zapisu zmian.
        """
        if self.image is not None and self.modified:
            if messagebox.askyesno("Zamykanie", "Czy chcesz zapisać obraz przed zamknięciem programu?"):
                self.save_image()
        self.root.destroy()
        plt.close('all')
        sys.exit(0)

    def show_about_program(self):
        """
        3.2. Wyświetla okno z informacjami o programie i autorze.
        """
        info = (
            "Aplikacja zbiorcza z ćwiczeń laboratoryjnych i projektu\n"
            "\n"
            "Tytuł projektu: Aplikacja do przetwarzania obrazów z funkcją kadrowania\n"
            "Autor: Oleh Buriakivskyi\n"
            "Numer albumu: 21569\n"
            "WIT grupa dziekańska: I06IO1\n"
            "\n"
            "Przedmiot: Algorytmy przetwarzania obrazów 2024/2025\n"
            "Prowadzący laboratoriumy: dr inż. Roszkowiak Łukasz\n"
            "Prowadzący wykłady: dr hab. Korzyńska Anna\n"
            "\n"
            "Wersja: 2.0\n"
            "Data ostatniej aktualizacji: 27.06.2025"
        )
        messagebox.showinfo("Informacja o programie", info)

    def convert_to_gray(self):
        """
        3.3.1. Konwertuje obraz kolorowy na szaroodcieniowy.
        """
        # Sprawdź, czy jest obraz do konwersji
        if self.image is None:
            return
        
        # Konwersja obrazu z BGR do skali szarości
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Utwórz wersję 3-kanałową do wyświetlania (potrzebne do rysowania kolorowych linii)
        self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)

        self.modified = True
        self.push_history()
        self.display_image()
        self.show_histogram()

    def split_rgb_channels(self):
        """
        3.3.2. Otwiera okno dialogowe do wyboru kanału (R, G lub B), a następnie wyświetla wybrany kanał jako obraz w skali szarości.
        """
        # Sprawdzenie, czy jest obraz do operacji
        if self.image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")
            return
        
        # Sprawdzenie, czy obraz jest kolorowy
        if len(self.image.shape) < 3:
            messagebox.showwarning("Błąd", "Ta operacja wymaga obrazu kolorowego (RGB).")
            return
        
        # Utworzenie okna dialogowego do wyboru kanału
        dialog = tk.Toplevel(self.root)
        dialog.title("Wybierz kanał RGB")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Wybierz kanał do wyświetlenia:").pack(pady=10)

        channel_var = tk.StringVar(value="R")

        tk.Radiobutton(dialog, text="Czerwony (R)", variable=channel_var, value="R").pack(anchor="w", padx=20)
        tk.Radiobutton(dialog, text="Zielony (G)", variable=channel_var, value="G").pack(anchor="w", padx=20)
        tk.Radiobutton(dialog, text="Niebieski (B)", variable=channel_var, value="B").pack(anchor="w", padx=20)

        def apply_choice():
            # Pobranie wybranego kanału i wyświetlenie go jako obraz szaroodcieniowy
            choice = channel_var.get()
            b, g, r = cv2.split(self.image)

            if choice == "R":
                selected_channel = r
            elif choice == "G":
                selected_channel = g
            else:  # "B"
                selected_channel = b
            
            self.push_history()
            self.gray_image = selected_channel
            self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)
            self.modified = True
            self.display_image()
            self.show_histogram()
            
            dialog.destroy()

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Zastosuj", command=apply_choice).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Anuluj", command=dialog.destroy).pack(side=tk.LEFT, padx=10)

        dialog.update_idletasks()
        dialog.geometry(f"+{self.root.winfo_x() + 100}+{self.root.winfo_y() + 100}")

    def convert_to_hsv_or_lab(self):
        """
        3.3.3. Otwiera okno dialogowe do wyboru przestrzeni barw (HSV lub Lab),
        a następnie wyświetla jeden z kanałów wybranej przestrzeni.
        """
        # Sprawdzenie, czy jest obraz do operacji
        if self.image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")
            return
        
        # Sprawdzenie, czy obraz jest kolorowy
        if len(self.image.shape) < 3:
            messagebox.showwarning("Błąd", "Ta operacja wymaga obrazu kolorowego (RGB).")
            return
        
        # Utworzenie okna dialogowego do wyboru przestrzeni barw
        dialog = tk.Toplevel(self.root)
        dialog.title("Wybierz przestrzeń barw")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Wybierz przestrzeń barw do konwersji:").pack(pady=10)

        space_var = tk.StringVar(value="HSV")

        tk.Radiobutton(dialog, text="HSV (Hue, Saturation, Value)", variable=space_var, value="HSV").pack(anchor="w", padx=20)
        tk.Radiobutton(dialog, text="Lab (Lightness, a, b)", variable=space_var, value="Lab").pack(anchor="w", padx=20)

        def apply_choice():
            # Konwersja do wybranej przestrzeni barw i wyświetlenie pierwszego kanału
            choice = space_var.get()
            if choice == "HSV":
                converted_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
                c1, _, _ = cv2.split(converted_img)
                selected_channel = c1  # Kanał H
            else:  # "Lab"
                converted_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2Lab)
                c1, _, _ = cv2.split(converted_img)
                selected_channel = c1  # Kanał L
            
            self.push_history()
            self.gray_image = selected_channel
            self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)
            self.modified = True
            self.display_image()
            self.show_histogram()
            dialog.destroy()

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Zastosuj", command=apply_choice).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Anuluj", command=dialog.destroy).pack(side=tk.LEFT, padx=10)

        dialog.update_idletasks()
        dialog.geometry(f"+{self.root.winfo_x() + 100}+{self.root.winfo_y() + 100}")

    def stretch_histogram_manual(self):
        """
        3.3.4. Własny algorytm rozciągania histogramu obrazu szaroodcieniowego.
        """
        if self.image is not None:
            # Konwersja do skali szarości, jeśli potrzeba
            if self.gray_image is None:
                self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            min_val = np.min(self.gray_image)
            max_val = np.max(self.gray_image)

            # Zabezpieczenie przed dzieleniem przez 0
            if max_val - min_val < 1:
                messagebox.showwarning("Uwaga", "Zakres jasności zbyt mały do rozciągnięcia.")
                return

            # Zapis do historii
            self.push_history()

            # Rozciąganie histogramu
            stretched = ((self.gray_image.astype(np.float32) - min_val) * (255.0 / (max_val - min_val)))
            stretched = np.clip(stretched, 0, 255).astype(np.uint8)

            self.gray_image = stretched
            self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)

            self.modified = True
            self.display_image()
            self.show_histogram()

    def equalize_histogram_manual(self):
        """
        3.3.5. Własny algorytm equalizacji histogramu obrazu szaroodcieniowego.
        """
        if self.image is not None:
            if self.gray_image is None:
                self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # konwersja do szarości jeśli potrzeba
            self.push_history()  # zapisz do historii
            hist, bins = np.histogram(self.gray_image.flatten(), 256, [0, 256])  # histogram
            cdf = hist.cumsum()  # dystrybuanta
            cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())  # normalizacja
            cdf_normalized = cdf_normalized.astype(np.uint8)
            img_equalized = cdf_normalized[self.gray_image]  # equalizacja
            self.gray_image = img_equalized  # aktualizacja obrazu szarości
            self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)  # aktualizacja obrazu wyświetlanego

            self.modified = True
            self.display_image()
            self.show_histogram()

    def negative_image(self):
        """
        3.3.6. Negacja obrazu szaroodcieniowego lub kolorowego.
        """
        if self.image is None:
            return
        if len(self.image.shape) == 3 and self.image.shape[2] == 3:
            # Obraz kolorowy
            self.image = 255 - self.image
            self.gray_image = None
        else:
            # Obraz szaroodcieniowy
            if self.gray_image is None:
                self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.gray_image = 255 - self.gray_image
            self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)

        self.modified = True
        self.push_history()
        self.display_image()
        self.show_histogram()

    def stretch_manual_dialog(self):
        """
        3.3.7. Okno dialogowe do ręcznego rozciągania zakresu jasności.
        """
        if self.image is None:
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Rozciąganie zakresu")

        tk.Label(dialog, text="Zakres wejściowy:").pack()
        p1_frame = tk.Frame(dialog)
        p1_frame.pack()
        tk.Label(p1_frame, text="p1:").pack(side=tk.LEFT)
        p1_var = tk.StringVar(value="0")
        tk.Entry(p1_frame, textvariable=p1_var, width=10).pack(side=tk.LEFT)
        p2_frame = tk.Frame(dialog)
        p2_frame.pack()
        tk.Label(p2_frame, text="p2:").pack(side=tk.LEFT)
        p2_var = tk.StringVar(value="255")
        tk.Entry(p2_frame, textvariable=p2_var, width=10).pack(side=tk.LEFT)

        tk.Label(dialog, text="Zakres wyjściowy:").pack()
        q3_frame = tk.Frame(dialog)
        q3_frame.pack()
        tk.Label(q3_frame, text="q3:").pack(side=tk.LEFT)
        q3_var = tk.StringVar(value="0")
        tk.Entry(q3_frame, textvariable=q3_var, width=10).pack(side=tk.LEFT)
        q4_frame = tk.Frame(dialog)
        q4_frame.pack()
        tk.Label(q4_frame, text="q4:").pack(side=tk.LEFT)
        q4_var = tk.StringVar(value="255")
        tk.Entry(q4_frame, textvariable=q4_var, width=10).pack(side=tk.LEFT)

        def apply():
            try:
                p1 = int(p1_var.get())  # początek zakresu wejściowego
                p2 = int(p2_var.get())  # koniec zakresu wejściowego
                q3 = int(q3_var.get())  # początek zakresu wyjściowego
                q4 = int(q4_var.get())  # koniec zakresu wyjściowego
                if p1 >= p2 or q3 >= q4:
                    raise ValueError("Nieprawidłowe wartości zakresów")  # sprawdzenie poprawności
                self.stretch_manual(p1, p2, q3, q4)  # wywołanie rozciągania
                dialog.destroy()  # zamknięcie okna
            except ValueError as e:
                messagebox.showerror("Błąd", str(e))  # obsługa błędów

        tk.Button(dialog, text="Zastosuj", command=apply).pack(pady=10)  # przycisk zastosuj

        dialog.update_idletasks()
        dialog.geometry(f"{dialog.winfo_reqwidth()}x{dialog.winfo_reqheight()}")  # ustawienie rozmiaru okna

    def stretch_manual(self, p1, p2, q3, q4):
        """
        3.3.7. Rozciąga zakres wartości obrazu z [p1,p2] do [q3,q4].
        """
        if self.image is None:
            return
        if self.gray_image is None:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # konwersja do szarości jeśli potrzeba

        result = np.zeros_like(self.gray_image)  # nowy obraz wynikowy

        if p2 - p1 == 0:
            messagebox.showerror("Błąd", "Nieprawidłowy zakres (p1 == p2)")  # zabezpieczenie
            return

        for y in range(self.gray_image.shape[0]):
            for x in range(self.gray_image.shape[1]):
                pixel = float(self.gray_image[y, x])
                if pixel < p1:
                    result[y, x] = q3  # poniżej zakresu wejściowego
                elif pixel > p2:
                    result[y, x] = q4  # powyżej zakresu wejściowego
                else:
                    value = q3 + (pixel - p1) * (q4 - q3) / (p2 - p1)  # przeskalowanie
                    result[y, x] = np.clip(int(value), 0, 255)

        self.gray_image = result  # aktualizacja obrazu szarości
        self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)  # aktualizacja obrazu wyświetlanego

        self.modified = True
        self.push_history()
        self.display_image()
        self.show_histogram()

    def posterize(self, levels):
        """
        3.4.1. Redukcja poziomów szarości przez posteryzację.
        """
        if self.image is None:
            return
        if self.gray_image is None:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # konwersja do szarości jeśli potrzeba
        step = 256 // levels  # wyznaczenie kroku poziomu
        posterized = (self.gray_image // step) * step  # posteryzacja
        self.gray_image = posterized  # aktualizacja obrazu szarości
        self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)  # aktualizacja obrazu wyświetlanego

        self.modified = True
        self.push_history()
        self.display_image()
        self.show_histogram()

    def blur_dialog(self):
        """
        3.4.2. Okno dialogowe do wyboru metody wygładzania.
        """
        method = simpledialog.askstring("Wygładzanie", "Wpisz metodę: 'blur' lub 'gaussian'")
        if method in ['blur', 'gaussian']:
            self.apply_blur(method)
        else:
            messagebox.showwarning("Błąd", f"Nieznana metoda: {method}")
            return

    def apply_blur(self, method):
        """
        3.4.2 Wygładzanie obrazu metodą blur lub gaussianBlur.
        """
        if self.image is None:
            return
        if self.gray_image is None:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        if method == 'blur':
            blurred = cv2.blur(self.gray_image, (5, 5))
        else:  # gauss
            blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
        self.gray_image = blurred
        self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)
        
        self.modified = True
        self.push_history()
        self.display_image()
        self.show_histogram()

    def edge_detection_dialog(self):
        """
        3.4.3. Okno dialogowe do wyboru metody detekcji krawędzi.
        """
        method = simpledialog.askstring("Detekcja krawędzi", "Metoda: 'sobel', 'laplacian', 'canny'")
        if method:
            self.edge_detection(method)
        else:
            messagebox.showwarning("Błąd", f"Nieznana metoda: {method}")
            return

    def apply_edge_detection(self, method):
        """
        3.4.3. Detekcja krawędzi: Sobel, Laplacian, Canny.
        """
        if self.image is None:
            return
        if self.gray_image is None:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        if method == 'sobel':
            sobelx = cv2.Sobel(self.gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(self.gray_image, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elif method == 'laplacian':
            edges = cv2.Laplacian(self.gray_image, cv2.CV_64F)
            edges = np.abs(edges)  # bierzemy wartość bezwzględną
            edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
            edges = edges.astype(np.uint8)
        else:  # canny
            edges = cv2.Canny(self.gray_image, 100, 200)
        self.gray_image = edges
        self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)
        
        self.modified = True
        self.push_history()
        self.display_image()
        self.show_histogram()

    def apply_sharpen(self, mask_type):
        """
        3.4.4. Wyostrza obraz używając jednej z trzech masek konwolucyjnych.
        Dostępne maski: 1 (standardowa), 2 (silniejsza), 3 (najsilniejsza).
        """
        if self.image is None:
            return
        if self.gray_image is None:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        if mask_type == 1:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        elif mask_type == 2:
            kernel = np.array([[-2,-2,-2], [-2,17,-2], [-2,-2,-2]])
        else:  # mask_type == 3
            kernel = np.array([[-3,-3,-3], [-3,25,-3], [-3,-3,-3]])
        sharpened = cv2.filter2D(self.gray_image, -1, kernel)
        self.gray_image = sharpened
        self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)
        self.modified = True
        self.push_history()
        self.display_image()
        self.show_histogram()

    def prewitt_dialog(self):
        """
        3.4.5. Okno dialogowe do wyboru kierunku maski Prewitta.
        """
        direction = simpledialog.askinteger("Prewitt", "Kierunek (1-8):", minvalue=1, maxvalue=8)
        if direction:
            self.apply_prewitt(direction)

    def apply_prewitt(self, direction):
        """
        3.4.5. Detekcja krawędzi maską Prewitta w wybranym kierunku.
        """
        if self.image is None:
            return
        if self.gray_image is None:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        angle = angles[direction - 1]
        if angle == 0:
            kernel = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
        elif angle == 45:
            kernel = np.array([[0,1,1], [-1,0,1], [-1,-1,0]])
        elif angle == 90:
            kernel = np.array([[-1,-1,-1], [0,0,0], [1,1,1]])
        elif angle == 135:
            kernel = np.array([[-1,-1,0], [-1,0,1], [0,1,1]])
        elif angle == 180:
            kernel = np.array([[1,0,-1], [1,0,-1], [1,0,-1]])
        elif angle == 225:
            kernel = np.array([[0,-1,-1], [1,0,-1], [1,1,0]])
        elif angle == 270:
            kernel = np.array([[1,1,1], [0,0,0], [-1,-1,-1]])
        else:  # 315
            kernel = np.array([[1,1,0], [1,0,-1], [0,-1,-1]])
        edges = cv2.filter2D(self.gray_image, -1, kernel)
        self.gray_image = edges
        self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)

        self.modified = True
        self.push_history()
        self.display_image()
        self.show_histogram()

    def custom_mask_dialog(self):
        """
        3.4.6. Okno dialogowe do wprowadzenia własnej maski 3x3 i trybu brzegowego.
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Własna maska 3x3")
        dialog.transient(self.root)
        dialog.grab_set()

        entries = []
        for i in range(3):
            row = []
            for j in range(3):
                entry = tk.Entry(dialog, width=5)
                entry.insert(0, "0")
                entry.grid(row=i, column=j, padx=5, pady=5)
                row.append(entry)
            entries.append(row)

        border_types = ["constant", "replicate", "reflect", "wrap"]
        border_var = tk.StringVar(value=border_types[0])
        border_frame = tk.Frame(dialog)
        border_frame.grid(row=3, column=0, columnspan=3, pady=10)
        tk.Label(border_frame, text="Typ brzegu:").pack(side=tk.LEFT)
        for bt in border_types:
            tk.Radiobutton(border_frame, text=bt, variable=border_var, value=bt).pack(side=tk.LEFT)

        def on_submit():
            try:
                kernel = np.array([[float(entries[i][j].get()) for j in range(3)] for i in range(3)])
                border_type = border_var.get()
                self.apply_custom_mask(kernel, border_type)
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Błąd", "Wprowadź poprawne wartości liczbowe.")

        tk.Button(dialog, text="Zastosuj", command=on_submit).grid(row=4, column=0, columnspan=3, pady=10)

        dialog.update_idletasks()
        dialog.geometry(f"{dialog.winfo_reqwidth()}x{dialog.winfo_reqheight()}")

    def apply_custom_mask(self, kernel, border_type):
        """
        3.4.6. Zastosowanie własnej maski 3x3 z wybranym trybem brzegowym.
        """
        if self.image is None:
            return
        if self.gray_image is None:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        border = self.get_border_type(border_type)
        filtered = cv2.filter2D(self.gray_image, -1, kernel, borderType=border)
        self.gray_image = filtered
        self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)

        self.modified = True
        self.push_history()
        self.display_image()
        self.show_histogram()

    def median_filter_dialog(self):
        """
        3.4.7. Okno dialogowe do wyboru rozmiaru i trybu brzegowego filtru medianowego.
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Filtracja medianowa")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Rozmiar filtra:").grid(row=0, column=0, padx=5, pady=5)
        size_var = tk.StringVar(value="3")
        size_entry = tk.Entry(dialog, textvariable=size_var, width=5)
        size_entry.grid(row=0, column=1, padx=5, pady=5)

        border_types = ["constant", "replicate", "reflect", "wrap"]
        border_var = tk.StringVar(value=border_types[0])
        border_frame = tk.Frame(dialog)
        border_frame.grid(row=1, column=0, columnspan=2, pady=10)
        tk.Label(border_frame, text="Typ brzegu:").pack(side=tk.LEFT)
        for bt in border_types:
            tk.Radiobutton(border_frame, text=bt, variable=border_var, value=bt).pack(side=tk.LEFT)

        def apply():
            try:
                size = int(size_var.get())
                if size % 2 == 0:
                    raise ValueError("Rozmiar filtra musi być nieparzysty")
                border_type = border_var.get()
                self.apply_median_filter(size, border_type)
                dialog.destroy()
            except ValueError as e:
                messagebox.showerror("Błąd", str(e))

        tk.Button(dialog, text="Zastosuj", command=apply).grid(row=2, column=0, columnspan=2, pady=10)

        dialog.update_idletasks()
        dialog.geometry(f"{dialog.winfo_reqwidth()}x{dialog.winfo_reqheight()}")

    def apply_median_filter(self, size, mode):
        """
        3.4.7. Filtracja medianowa z obsługą brzegów (isolated, reflect, replicate).
        """
        if self.image is None:
            return
        if self.gray_image is None:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        border = self.get_border_type(mode)
        filtered = cv2.medianBlur(self.gray_image, size)
        self.gray_image = filtered
        self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)

        self.modified = True
        self.push_history()
        self.display_image()
        self.show_histogram()

    def apply_padding(self, img, size, mode):
        """
        3.4.7. Dodaje padding do obrazu zgodnie z wybranym trybem brzegowym.
        """
        if mode == "constant":
            return cv2.copyMakeBorder(img, size//2, size//2, size//2, size//2, cv2.BORDER_CONSTANT, value=0)
        elif mode == "replicate":
            return cv2.copyMakeBorder(img, size//2, size//2, size//2, size//2, cv2.BORDER_REPLICATE)
        elif mode == "reflect":
            return cv2.copyMakeBorder(img, size//2, size//2, size//2, size//2, cv2.BORDER_REFLECT)
        else:  # wrap
            return cv2.copyMakeBorder(img, size//2, size//2, size//2, size//2, cv2.BORDER_WRAP)

    def _binary_operation_helper(self, operation):
        """
        3.4.8. Pomocnicza funkcja do operacji dwuargumentowych (wczytuje drugi obraz).
        """
        if self.image is None:
            return
        if self.gray_image is None:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.bmp;*.jpg;*.png;*.gif")])
        if not path:
            return
        second_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if second_img is None:
            messagebox.showerror("Błąd", "Nie udało się wczytać drugiego obrazu.")
            return
        if self.gray_image is None:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        if second_img.shape != self.gray_image.shape:
            messagebox.showerror("Błąd", "Obrazy muszą mieć te same wymiary.")
            return
        self.apply_binary_operation(second_img, operation)

    def binary_add(self):
        """
        3.4.8. Dodawanie dwóch obrazów.
        """
        self._binary_operation_helper("add")

    def binary_subtract(self):
        """
        3.4.8. Odejmowanie dwóch obrazów.
        """
        self._binary_operation_helper("subtract")

    def binary_and(self):
        """
        3.4.8. Operacja AND na dwóch obrazach.
        """
        self._binary_operation_helper("and")

    def binary_or(self):
        """
        3.4.8. Operacja OR na dwóch obrazach.
        """
        self._binary_operation_helper("or")

    def binary_xor(self):
        """
        3.4.8. Operacja XOR na dwóch obrazach.
        """
        self._binary_operation_helper("xor")

    def binary_blend_dialog(self):
        """
        3.4.8. Okno dialogowe do mieszania dwóch obrazów.
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Mieszanie obrazów")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Współczynnik mieszania (0-1):").pack(pady=5)
        alpha_var = tk.DoubleVar(value=0.5)
        alpha_scale = tk.Scale(dialog, from_=0, to=1, resolution=0.1, orient=tk.HORIZONTAL, variable=alpha_var)
        alpha_scale.pack(fill=tk.X, padx=10)

        def choose_and_blend():
            if self.gray_image is None:
                self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            path = filedialog.askopenfilename(filetypes=[("Image files", "*.bmp;*.jpg;*.png;*.gif")])
            if path:
                second_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if second_img is not None:
                    if self.gray_image is None:
                        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                    if second_img.shape == self.gray_image.shape:
                        self.apply_binary_operation(second_img, "blend", alpha_var.get())
                        dialog.destroy()
                    else:
                        messagebox.showerror("Błąd", "Obrazy muszą mieć te same wymiary.")
                else:
                    messagebox.showerror("Błąd", "Nie udało się wczytać drugiego obrazu.")

        tk.Button(dialog, text="Wybierz obraz i mieszaj", command=choose_and_blend).pack(pady=10)

        dialog.update_idletasks()
        dialog.geometry(f"{dialog.winfo_reqwidth()}x{dialog.winfo_reqheight()}")

    def apply_binary_operation(self, second_img, operation, alpha=0.5):
        """
        3.4.8. Wykonuje wybraną operację dwuargumentową na obrazach.
        """
        if self.image is None or self.gray_image is None:
            return
        if operation == "add":
            result = cv2.add(self.gray_image, second_img)
        elif operation == "subtract":
            result = cv2.subtract(self.gray_image, second_img)
        elif operation == "and":
            result = cv2.bitwise_and(self.gray_image, second_img)
        elif operation == "or":
            result = cv2.bitwise_or(self.gray_image, second_img)
        elif operation == "xor":
            result = cv2.bitwise_xor(self.gray_image, second_img)
        elif operation == "blend":
            result = cv2.addWeighted(self.gray_image, 1-alpha, second_img, alpha, 0)
        else:
            return

        self.gray_image = result
        self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)
        
        self.modified = True
        self.push_history()
        self.display_image()
        self.show_histogram()

    def run_two_vs_one_step_filter(self, blur_choice, sharp_choice):
        """
        3.4.9. Porównanie filtracji dwuetapowej i jednoetapowej (maska 5x5).
        """
        if self.image is None:
            return
        if self.gray_image is None:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Filtracja dwuetapowa
        if blur_choice == 'blur':
            blurred = cv2.blur(self.gray_image, (5, 5))
        else:  # gauss
            blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)

        if sharp_choice == 1:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        elif sharp_choice == 2:
            kernel = np.array([[-2,-2,-2], [-2,17,-2], [-2,-2,-2]])
        else:  # sharp_choice == 3
            kernel = np.array([[-3,-3,-3], [-3,25,-3], [-3,-3,-3]])

        two_step = cv2.filter2D(blurred, -1, kernel)

        # Filtracja jednoetapowa
        if blur_choice == 'blur':
            one_step_kernel = cv2.filter2D(np.ones((5,5))/25, -1, kernel)
        else:  # gauss
            gauss_kernel = cv2.getGaussianKernel(5, 0)
            gauss_kernel = gauss_kernel * gauss_kernel.T
            one_step_kernel = cv2.filter2D(gauss_kernel, -1, kernel)

        one_step = cv2.filter2D(self.gray_image, -1, one_step_kernel)

        # Wyświetlanie wyników
        plt.close('all')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(two_step, cmap='gray')
        ax1.set_title('Filtracja dwuetapowa')
        ax1.axis('off')
        ax2.imshow(one_step, cmap='gray')
        ax2.set_title('Filtracja jednoetapowa')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()

    def compare_two_step_and_one_step_filter(self):
        """
        3.4.9. Okno dialogowe do wyboru masek do filtracji dwuetapowej/jednoetapowej.
        """
        dialog = tk.Toplevel(self.root)  # Tworzenie nowego okna dialogowego
        dialog.title("Porównanie filtracji")  # Ustawienie tytułu okna
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Typ wygładzania:").pack(pady=5)  # Etykieta dla wyboru typu wygładzania
        blur_var = tk.StringVar(value="blur")  # Zmienna do przechowywania wyboru typu wygładzania
        tk.Radiobutton(dialog, text="Średnia (Blur)", variable=blur_var, value="blur").pack()  # Opcja: Średnia
        tk.Radiobutton(dialog, text="Gaussowskie", variable=blur_var, value="gauss").pack()  # Opcja: Gaussowskie

        tk.Label(dialog, text="Maska wyostrzająca:").pack(pady=5)  # Etykieta dla wyboru maski wyostrzającej
        sharp_var = tk.IntVar(value=1)  # Zmienna do przechowywania wyboru maski wyostrzającej
        tk.Radiobutton(dialog, text="Maska 1", variable=sharp_var, value=1).pack()  # Opcja: Maska 1
        tk.Radiobutton(dialog, text="Maska 2", variable=sharp_var, value=2).pack()  # Opcja: Maska 2
        tk.Radiobutton(dialog, text="Maska 3", variable=sharp_var, value=3).pack()  # Opcja: Maska 3

        def apply_filter():
            self.run_two_vs_one_step_filter(blur_var.get(), sharp_var.get())  # Wywołanie funkcji porównującej filtry z wybranymi opcjami
            dialog.destroy()  # Zamknięcie okna dialogowego

        tk.Button(dialog, text="Porównaj", command=apply_filter).pack(pady=10)  # Przycisk uruchamiający porównanie

        dialog.update_idletasks()
        dialog.geometry(f"{dialog.winfo_reqwidth()}x{dialog.winfo_reqheight()}")  # Ustawienie rozmiaru okna na wymagany

    def morphological_operation(self, operation_type):
        """
        3.5.1. Operacje morfologiczne: erozja, dylatacja, otwarcie, zamknięcie.
        Obsługa elementów strukturalnych (romb, kwadrat) i trybów brzegowych.
        """
        if self.image is None:
            return
        if self.gray_image is None:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # Konwersja do szarości jeśli potrzeba

        dialog = tk.Toplevel(self.root)  # Tworzenie okna dialogowego
        dialog.title(f"Operacja morfologiczna: {operation_type}")  # Ustawienie tytułu
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Rozmiar elementu strukturalnego:").pack(pady=5)  # Etykieta rozmiaru
        size_var = tk.StringVar(value="3")  # Zmienna na rozmiar elementu
        size_entry = tk.Entry(dialog, textvariable=size_var, width=5)  # Pole do wpisania rozmiaru
        size_entry.pack()

        tk.Label(dialog, text="Typ elementu strukturalnego:").pack(pady=5)  # Etykieta typu elementu
        shape_var = tk.StringVar(value="rect")  # Zmienna na typ elementu
        tk.Radiobutton(dialog, text="Prostokąt", variable=shape_var, value="rect").pack()  # Opcja: prostokąt
        tk.Radiobutton(dialog, text="Krzyż", variable=shape_var, value="cross").pack()  # Opcja: krzyż
        tk.Radiobutton(dialog, text="Elipsa", variable=shape_var, value="ellipse").pack()  # Opcja: elipsa

        def apply():
            try:
                size = int(size_var.get())  # Pobranie rozmiaru
                if size % 2 == 0:
                    raise ValueError("Rozmiar elementu strukturalnego musi być nieparzysty")  # Sprawdzenie nieparzystości
                shape = shape_var.get()  # Pobranie typu elementu
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT if shape == "rect" else
                    cv2.MORPH_CROSS if shape == "cross" else
                    cv2.MORPH_ELLIPSE,
                    (size, size)
                )  # Utworzenie elementu strukturalnego
                if operation_type == "Erozja":
                    result = cv2.erode(self.gray_image, kernel)  # Erozja
                elif operation_type == "Dylatacja":
                    result = cv2.dilate(self.gray_image, kernel)  # Dylatacja
                elif operation_type == "Otwarcie":
                    result = cv2.morphologyEx(self.gray_image, cv2.MORPH_OPEN, kernel)  # Otwarcie
                else:  # Zamknięcie
                    result = cv2.morphologyEx(self.gray_image, cv2.MORPH_CLOSE, kernel)
                
                self.gray_image = result
                self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)

                self.modified = True
                self.push_history()
                self.display_image()
                self.show_histogram()
                dialog.destroy()
            except ValueError as e:
                messagebox.showerror("Błąd", str(e))  # Obsługa błędów

        tk.Button(dialog, text="Zastosuj", command=apply).pack(pady=10)  # Przycisk zastosowania operacji

        dialog.update_idletasks()
        dialog.geometry(f"{dialog.winfo_reqwidth()}x{dialog.winfo_reqheight()}")  # Ustawienie rozmiaru okna

    def get_border_type(self, mode):
        """
        3.5.1. Zwraca odpowiedni typ brzegu dla OpenCV na podstawie wybranego trybu.
        """
        if mode == "constant":
            return cv2.BORDER_CONSTANT  # Stała wartość na brzegu
        elif mode == "replicate":
            return cv2.BORDER_REPLICATE  # Powielanie pikseli brzegowych
        elif mode == "reflect":
            return cv2.BORDER_REFLECT  # Odbicie lustrzane na brzegu
        else:  # wrap
            return cv2.BORDER_WRAP  # Zawijanie obrazu na brzegu

    def thinning_dialog(self):
        """
        3.5.2. Okno dialogowe do szkieletyzacji obrazu metodą Zhang-Suen.
        """
        dialog = tk.Toplevel(self.root)  # Tworzenie nowego okna dialogowego
        dialog.title("Szkieletyzacja Zhang-Suen")  # Ustawienie tytułu okna
        dialog.transient(self.root)  # Okno jest powiązane z głównym oknem
        dialog.grab_set()  # Przechwycenie fokusu przez okno dialogowe

        tk.Label(dialog, text="Typ brzegu:").pack(pady=5)  # Etykieta informująca o wyborze typu brzegu
        border_var = tk.StringVar(value="constant")  # Zmienna przechowująca wybrany typ brzegu
        tk.Radiobutton(dialog, text="Constant", variable=border_var, value="constant").pack()  # Opcja "Constant"
        tk.Radiobutton(dialog, text="Replicate", variable=border_var, value="replicate").pack()  # Opcja "Replicate"
        tk.Radiobutton(dialog, text="Reflect", variable=border_var, value="reflect").pack()  # Opcja "Reflect"
        tk.Radiobutton(dialog, text="Wrap", variable=border_var, value="wrap").pack()  # Opcja "Wrap"

        def apply_thinning():
            self.thinning_zhang_suen(border_var.get())  # Wywołanie funkcji szkieletyzacji z wybranym typem brzegu
            dialog.destroy()  # Zamknięcie okna dialogowego

        tk.Button(dialog, text="Zastosuj", command=apply_thinning).pack(pady=10)  # Przycisk do zastosowania operacji

        dialog.update_idletasks()  # Aktualizacja rozmiaru okna
        dialog.geometry(f"{dialog.winfo_reqwidth()}x{dialog.winfo_reqheight()}")  # Ustawienie rozmiaru okna na wymagany

    def thinning_zhang_suen(self, border_mode="constant"):
        """
        3.5.2. Szkieletyzacja (algorytm Zhang-Suen) z obsługą trybów brzegowych.
        """
        if self.image is None:
            return
        if self.gray_image is None:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Konwersja do obrazu binarnego
        _, binary = cv2.threshold(self.gray_image, 127, 255, cv2.THRESH_BINARY)  # Progowanie Otsu
        binary = binary // 255  # Konwersja do 0/1

        def neighbours(padded, x, y):
            # Zwraca sąsiadów piksela (x,y) w kolejności zgodnej z algorytmem Zhang-Suen.
            return [
                padded[y-1, x], padded[y-1, x+1], padded[y, x+1], padded[y+1, x+1],
                padded[y+1, x], padded[y+1, x-1], padded[y, x-1], padded[y-1, x-1]
            ]

        def transitions(neigh):
            #Liczy liczbę przejść 0->1 w sąsiedztwie.
            return sum((neigh[i] == 0 and neigh[(i+1)%8] == 1) for i in range(8))

        # Dodanie paddingu z odpowiednim typem brzegu
        border = self.get_border_type(border_mode)
        if border_mode == "constant":
            padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, border, value=0)  # Padding zerami
        else:
            padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, border)  # Padding zgodnie z trybem
        height, width = padded.shape

        while True:
            # Pierwszy podkrok algorytmu
            to_remove = set()
            for y in range(1, height-1):
                for x in range(1, width-1):
                    if padded[y, x] == 1:
                        neigh = neighbours(padded, x, y)
                        # Sprawdzenie warunków usuwania piksela
                        if (2 <= sum(neigh) <= 6 and
                            transitions(neigh) == 1 and
                            neigh[0] * neigh[2] * neigh[4] == 0 and
                            neigh[2] * neigh[4] * neigh[6] == 0):
                            to_remove.add((x, y))
            if not to_remove:
                break
            for x, y in to_remove:
                padded[y, x] = 0  # Usuwanie pikseli

            # Drugi podkrok algorytmu
            to_remove = set()
            for y in range(1, height-1):
                for x in range(1, width-1):
                    if padded[y, x] == 1:
                        neigh = neighbours(padded, x, y)
                        # Sprawdzenie warunków usuwania piksela (drugi podkrok)
                        if (2 <= sum(neigh) <= 6 and
                            transitions(neigh) == 1 and
                            neigh[0] * neigh[2] * neigh[6] == 0 and
                            neigh[0] * neigh[4] * neigh[6] == 0):
                            to_remove.add((x, y))
            if not to_remove:
                break
            for x, y in to_remove:
                padded[y, x] = 0  # Usuwanie pikseli

        # Usunięcie paddingu i konwersja z powrotem do 0-255
        result = padded[1:-1, 1:-1] * 255  # Usunięcie obramowania
        self.gray_image = result.astype(np.uint8)  # Aktualizacja obrazu szarości
        self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)  # Aktualizacja obrazu kolorowego
        
        self.modified = True
        self.push_history()
        self.display_image()
        self.show_histogram()

    def detect_lines_hough(self):
        """
        3.5.3. Detekcja linii prostych za pomocą transformaty Hougha.
        """
        if self.image is None:
            return
        if self.gray_image is None:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Wykrywanie krawędzi
        edges = cv2.Canny(self.gray_image, 50, 150)

        # Wykrywanie linii
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

        # Rysowanie linii
        result = self.image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Wyświetlanie wyników
        plt.close('all')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Oryginalny obraz')
        ax1.axis('off')
        ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        ax2.set_title('Wykryte linie')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()

    def hough_parameters_dialog(self):
        """
        3.5.3. Okno dialogowe do ustawienia parametrów transformaty Hougha.
        """
        dialog = tk.Toplevel(self.root)  # Tworzenie nowego okna dialogowego
        dialog.title("Parametry transformaty Hougha")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Próg detekcji krawędzi:").pack(pady=5)
        edge_thresh_var = tk.StringVar(value="50")  # Zmienna progowa dla detekcji krawędzi
        tk.Entry(dialog, textvariable=edge_thresh_var, width=10).pack()

        tk.Label(dialog, text="Próg detekcji linii:").pack(pady=5)
        line_thresh_var = tk.StringVar(value="100")  # Zmienna progowa dla detekcji linii
        tk.Entry(dialog, textvariable=line_thresh_var, width=10).pack()

        tk.Label(dialog, text="Minimalna długość linii:").pack(pady=5)
        min_length_var = tk.StringVar(value="100")  # Minimalna długość wykrywanej linii
        tk.Entry(dialog, textvariable=min_length_var, width=10).pack()

        tk.Label(dialog, text="Maksymalna przerwa:").pack(pady=5)
        max_gap_var = tk.StringVar(value="10")  # Maksymalna przerwa między segmentami linii
        tk.Entry(dialog, textvariable=max_gap_var, width=10).pack()

        def apply():
            try:
                edge_thresh = int(edge_thresh_var.get())  # Pobranie progu detekcji krawędzi
                line_thresh = int(line_thresh_var.get())  # Pobranie progu detekcji linii
                min_length = int(min_length_var.get())    # Pobranie minimalnej długości linii
                max_gap = int(max_gap_var.get())          # Pobranie maksymalnej przerwy
                if self.image is None:
                    return
                if self.gray_image is None:
                    self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # Konwersja do skali szarości
                edges = cv2.Canny(self.gray_image, edge_thresh, edge_thresh*3)  # Detekcja krawędzi
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=line_thresh,
                                      minLineLength=min_length, maxLineGap=max_gap)  # Wykrywanie linii Hougha
                result = self.image.copy()  # Kopia obrazu do rysowania linii
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Rysowanie wykrytych linii
                plt.close('all')
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # Przygotowanie dwóch wykresów
                ax1.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
                ax1.set_title('Oryginalny obraz')
                ax1.axis('off')
                ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                ax2.set_title('Wykryte linie')
                ax2.axis('off')
                plt.tight_layout()
                plt.show()  # Wyświetlenie wyników
                dialog.destroy()  # Zamknięcie okna dialogowego
            except ValueError as e:
                messagebox.showerror("Błąd", str(e))  # Obsługa błędów konwersji

        tk.Button(dialog, text="Zastosuj", command=apply).pack(pady=10)  # Przycisk do zastosowania parametrów

        dialog.update_idletasks()
        dialog.geometry(f"{dialog.winfo_reqwidth()}x{dialog.winfo_reqheight()}")  # Ustawienie rozmiaru okna

    def canvas_bindings(self):
        """
        3.5.4. Ustawia bindowania zdarzeń na canvasie (np. do linii profilu).
        """
        pass
        self.img_panel_container.bind("<B1-Motion>", self.profile_line_drag)
        self.img_panel_container.bind("<ButtonRelease-1>", self.profile_line_release)

    def start_profile_selection(self):
        """
        3.5.4. Rozpoczyna wybór linii profilu na obrazie (2 punkty).
        """
        if self.image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")  # Sprawdzenie czy obraz jest wczytany
            return

        # Uwaga jeśli obraz nie jest szaroodcieniowy
        if self.gray_image is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz w skali szarości.")  # Sprawdzenie czy obraz jest w skali szarości
            return

        # Reset punktów profilu
        self.profile_points = []  # Wyczyszczenie poprzednich punktów profilu
        
        # Bindowanie kliknięcia myszy
        self.img_panel_container.bind("<Button-1>", self.on_profile_click)  # Przypisanie obsługi kliknięcia do wyboru punktów
        
        messagebox.showinfo("Linia profilu", "Kliknij w dwóch miejscach, aby wyznaczyć linię profilu.")  # Instrukcja dla użytkownika

    def on_profile_click(self, event):
        """
        3.5.4. Obsługuje kliknięcia do wyznaczania linii profilu.
        """
        x = int((event.x - self.image_position[0]) / self.zoom_factor)
        y = int((event.y - self.image_position[1]) / self.zoom_factor)
        
        if len(self.profile_points) >= 2:
            self.profile_points = []
            
        self.profile_points.append((x, y))

        if len(self.profile_points) == 2:
            # Rysuj linię na kopii aktualnego obrazu
            img_with_line = self.image.copy()
            cv2.line(img_with_line, 
                    self.profile_points[0],
                    self.profile_points[1],
                    (0, 255, 0), 3, cv2.LINE_AA)
            
            # Aktualizuj wyświetlany obraz
            self.image = img_with_line
            self.display_image()
            
            # Pokaż profil
            self.plot_profile()

    def plot_profile(self):
        """
        3.5.4. Rysuje wykres profilu intensywności wzdłuż wybranej linii.
        """
        if len(self.profile_points) != 2:
            return

        x0, y0 = self.profile_points[0]
        x1, y1 = self.profile_points[1]
        length = int(np.hypot(x1 - x0, y1 - y0))
        x_vals = np.linspace(x0, x1, length).astype(np.int32)
        y_vals = np.linspace(y0, y1, length).astype(np.int32)

        # Zabezpieczenie przed wyjściem poza granice obrazu
        x_vals = np.clip(x_vals, 0, self.gray_image.shape[1] - 1)
        y_vals = np.clip(y_vals, 0, self.gray_image.shape[0] - 1)

        gray = self.gray_image if self.gray_image is not None else cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        values = gray[y_vals, x_vals]

        # Zamykamy wszystkie istniejące wykresy
        plt.close('all')
        
        # Tworzenie nowego okna z wykresem
        fig = plt.figure("Profil intensywności")
        plt.plot(values, color="black")
        plt.title("Profil intensywności pikseli")
        plt.xlabel("Odległość wzdłuż linii")
        plt.ylabel("Wartość intensywności")
        plt.grid(True)
        
        # Wyłączenie interaktywnych funkcji
        plt.gca().set_position([0.1, 0.1, 0.85, 0.85])
        
        # Dodanie obsługi zamknięcia okna
        def on_close(event):
            self.restore_image()
        fig.canvas.mpl_connect('close_event', on_close)
        
        plt.show(block=False)

    def restore_image(self):
        """
        3.5.4. Przywraca obraz z historii (po zamknięciu okna profilu).
        """
        if len(self.history) > 0:
            self.image = self.history[-1][0].copy()
            self.display_image()

    def profile_line_drag(self, event):
        """
        3.5.4. Obsługuje przeciąganie linii profilu.
        """
        # Sprawdzenie, czy linia profilu jest aktywna i w trybie przesuwania
        if self.profile_start and self.profile_end and self.profile_moving:
            # Oblicz przesunięcie względem poprzedniej pozycji
            dx = int(event.x / self.zoom_factor) - self.move_start[0]  # zmiana X
            dy = int(event.y / self.zoom_factor) - self.move_start[1]  # zmiana Y
            # Przesuń oba końce linii o wyliczone przesunięcie
            self.profile_start = (self.profile_start[0] + dx, self.profile_start[1] + dy)
            self.profile_end = (self.profile_end[0] + dx, self.profile_end[1] + dy)
            # Zaktualizuj punkt początkowy przesuwania
            self.move_start = (int(event.x / self.zoom_factor), int(event.y / self.zoom_factor))
            # Zaktualizuj pozycję linii na canvasie
            self.update_profile_line()
            # Narysuj ponownie wykres profilu
            self.draw_profile_plot()

    def profile_line_release(self, event):
        """
        3.5.4. Kończy przeciąganie linii profilu.
        """
        # Wyłącz tryb przesuwania linii profilu
        self.profile_moving = False

    def update_profile_line(self):
        """
        3.5.4. Aktualizuje pozycję linii profilu na canvasie.
        """
        # Sprawdź, czy oba końce linii profilu są ustawione
        if self.profile_start and self.profile_end:
            # Usuń poprzednią linię z canvasu, jeśli istnieje
            if self.profile_line:
                self.img_panel_container.delete(self.profile_line)
            x0, y0 = self.profile_start
            x1, y1 = self.profile_end
            # Narysuj nową linię na canvasie
            self.profile_line = self.img_panel_container.create_line(
                x0 * self.zoom_factor, y0 * self.zoom_factor,
                x1 * self.zoom_factor, y1 * self.zoom_factor,
                fill="#00FF00", width=2
            )

    def draw_profile_line(self):
        """
        3.5.4. Rysuje linię profilu na canvasie.
        """
        # Usuń poprzednią linię profilu, jeśli istnieje
        if self.profile_line:
            self.img_panel_container.delete(self.profile_line)

        x0, y0 = self.profile_points[0]
        x1, y1 = self.profile_points[1]

        # Narysuj nową linię profilu na canvasie
        self.profile_line = self.img_panel_container.create_line(
            x0 * self.zoom_factor, y0 * self.zoom_factor,
            x1 * self.zoom_factor, y1 * self.zoom_factor,
            fill="#00FF00", width=2
        )

    def image_pyramid(self):
        """
        3.5.5. Tworzy i wyświetla piramidę obrazów (2 poziomy w górę i w dół).
        """
        if self.image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")
            return

        # Tworzenie piramidy
        pyramid = []
        current = self.image.copy()
        smaller_levels = []
        # 2 poziomy w górę (zmniejszanie)
        for i in range(2):
            h, w = current.shape[:2]
            smaller = cv2.pyrDown(current)
            smaller_levels.append((f"Poziom -{i+1}", smaller))
            current = smaller
        smaller_levels = smaller_levels[::-1]  # Odwróć kolejność: -2, -1

        # Obraz oryginalny
        pyramid.append(("Oryginalny", self.image))
        current = self.image.copy()

        # 2 poziomy w dół (zwiększanie)
        for i in range(2):
            h, w = current.shape[:2]
            larger = cv2.pyrUp(current)
            pyramid.append((f"Poziom +{i+1}", larger))
            current = larger

        # Połącz listy: najpierw -2, -1, potem oryginalny, potem +1, +2
        pyramid = smaller_levels + pyramid

        # Wyświetlanie wyników
        pyramid_window = tk.Toplevel(self.root)
        pyramid_window.title("Piramida obrazów")
        
        # Pobierz rozmiar ekranu
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Stwórz główny kontener ze scrollbarami
        main_container = ttk.Frame(pyramid_window)
        main_container.pack(fill="both", expand=True)

        # Dodaj Canvas i Scrollbary
        canvas = tk.Canvas(main_container)
        v_scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(main_container, orient="horizontal", command=canvas.xview)
        
        # Skonfiguruj scrollbary
        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Połącz canvas ze scrollbarami
        canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)

        # Stwórz ramkę wewnątrz canvasa
        inner_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")

        # Ustaw większy rozmiar dla pojedynczego obrazu
        base_image_width = 600  # Zwiększony bazowy rozmiar obrazu
        
        # Organizacja okna w siatkę
        for i, (level_name, img) in enumerate(pyramid):
            # Stwórz ramkę dla każdego poziomu
            frame = ttk.LabelFrame(inner_frame, text=level_name)
            frame.grid(row=i//2, column=i%2, padx=10, pady=10, sticky="nsew")
            
            # Kontener na obraz i etykietę
            container = ttk.Frame(frame)
            container.pack(expand=True, fill="both")
            
            # Skalowanie obrazu zachowując proporcje
            h, w = img.shape[:2]
            scale = base_image_width / w
            new_size = (int(w * scale), int(h * scale))
            
            img_resized = cv2.resize(img, new_size)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
            
            # Kontener na obraz
            img_frame = ttk.Frame(container)
            img_frame.pack(expand=True, fill="both", padx=5, pady=5)
            
            # Wyświetl obraz
            label = tk.Label(img_frame, image=img_tk)
            label.image = img_tk
            label.pack(expand=True)

            # Dodaj informację o rozmiarze w osobnej ramce
            info_frame = ttk.Frame(container)
            info_frame.pack(fill="x", padx=5, pady=(0, 5))
            size_label = ttk.Label(info_frame, 
                                 text=f"Rozmiar: {img.shape[1]}x{img.shape[0]}", 
                                 anchor="center")
            size_label.pack(expand=True)

        # Wyrównaj kolumny i wiersze w inner_frame
        for i in range(3):  # 3 rzędy
            inner_frame.grid_rowconfigure(i, weight=1)
        for i in range(2):  # 2 kolumny
            inner_frame.grid_columnconfigure(i, weight=1)

        # Funkcja aktualizacji scrollregion
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        # Bind funkcji aktualizacji
        inner_frame.bind("<Configure>", on_frame_configure)

        # Ustaw rozmiar okna
        window_width = min(screen_width - 100, 1400)  # Maksymalna szerokość
        window_height = min(screen_height - 100, 900)  # Maksymalna wysokość
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        # Ustaw geometrię okna
        pyramid_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Dodaj obsługę scrollowania myszką
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", on_mousewheel)

    def manual_threshold(self):
        """
        3.6.1. Segmentacja progowa z ręcznym wyborem progu.
        """
        # Sprawdzenie, czy obraz został wczytany
        if self.image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")
            return

        # Tworzenie okna dialogowego do wyboru progu
        dialog = tk.Toplevel(self.root)
        dialog.title("Progowanie ręczne")
        dialog.grab_set()

        # Konwersja obrazu do skali szarości, jeśli jeszcze nie została wykonana
        if self.gray_image is None:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Utworzenie suwaka do wyboru wartości progu
        tk.Label(dialog, text="Próg:").pack(pady=5)
        threshold_var = tk.IntVar(value=128)
        threshold_slider = tk.Scale(dialog, from_=0, to=255, orient=tk.HORIZONTAL, 
                                  variable=threshold_var, length=200)
        threshold_slider.pack(pady=5)

        # Etykieta do podglądu progowanego obrazu
        preview_label = tk.Label(dialog)
        preview_label.pack(pady=10)

        def update_preview(*args):
            # Aktualizacja podglądu po zmianie wartości progu
            _, binary = cv2.threshold(self.gray_image, threshold_var.get(), 255, cv2.THRESH_BINARY)
            preview = cv2.resize(binary, (400, 400))
            img_tk = ImageTk.PhotoImage(Image.fromarray(preview))
            preview_label.configure(image=img_tk)
            preview_label.image = img_tk

        def apply():
            # Zastosowanie wybranego progu do obrazu i zamknięcie okna dialogowego
            _, binary = cv2.threshold(self.gray_image, threshold_var.get(), 255, cv2.THRESH_BINARY)
            self.push_history()
            self.image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            self.gray_image = binary
            self.display_image()
            dialog.destroy()

        # Połączenie suwaka z funkcją aktualizującą podgląd
        threshold_var.trace('w', update_preview)
        update_preview()

        # Przycisk do zatwierdzenia wyboru progu
        tk.Button(dialog, text="Zastosuj", command=apply).pack(pady=10)

        # Ustawienie rozmiaru okna dialogowego
        dialog.update_idletasks()
        dialog.geometry(f"{dialog.winfo_reqwidth()}x{dialog.winfo_reqheight()}")

    def adaptive_threshold(self):
        """
        3.6.1. Segmentacja progowa adaptacyjna (mean/gaussian).
        """
        # Sprawdzenie, czy obraz jest wczytany
        if self.image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")
            return

        # Tworzenie okna dialogowego do ustawień progowania adaptacyjnego
        dialog = tk.Toplevel(self.root)
        dialog.title("Progowanie adaptacyjne")
        dialog.grab_set()

        # Konwersja do skali szarości, jeśli potrzeba
        if self.gray_image is None:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Parametry progowania adaptacyjnego
        method_var = tk.StringVar(value="GAUSSIAN")  # Metoda progowania
        block_size_var = tk.IntVar(value=11)          # Rozmiar bloku
        c_var = tk.IntVar(value=2)                    # Stała C

        # Wybór metody progowania (GAUSSIAN lub MEAN)
        tk.Label(dialog, text="Metoda:").pack(pady=5)
        ttk.Combobox(dialog, textvariable=method_var, 
                    values=["GAUSSIAN", "MEAN"]).pack(pady=5)

        # Ustawienie rozmiaru bloku (musi być nieparzysty)
        tk.Label(dialog, text="Rozmiar bloku:").pack(pady=5)
        tk.Scale(dialog, from_=3, to=99, orient=tk.HORIZONTAL, 
                variable=block_size_var, resolution=2).pack(pady=5)

        # Ustawienie stałej C (odejmowana od średniej/gaussa)
        tk.Label(dialog, text="Stała C:").pack(pady=5)
        tk.Scale(dialog, from_=0, to=20, orient=tk.HORIZONTAL, 
                variable=c_var).pack(pady=5)

        # Etykieta do podglądu progowanego obrazu
        preview_label = tk.Label(dialog)
        preview_label.pack(pady=10)

        def update_preview(*args):
            # Aktualizacja podglądu po zmianie parametrów
            method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method_var.get() == "GAUSSIAN" else cv2.ADAPTIVE_THRESH_MEAN_C
            binary = cv2.adaptiveThreshold(self.gray_image, 255, method,
                                         cv2.THRESH_BINARY, block_size_var.get(), c_var.get())
            preview = cv2.resize(binary, (400, 400))
            img_tk = ImageTk.PhotoImage(Image.fromarray(preview))
            preview_label.configure(image=img_tk)
            preview_label.image = img_tk

        def apply():
            # Zastosowanie wybranych parametrów do obrazu i zamknięcie okna dialogowego
            method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method_var.get() == "GAUSSIAN" else cv2.ADAPTIVE_THRESH_MEAN_C
            binary = cv2.adaptiveThreshold(self.gray_image, 255, method,
                                         cv2.THRESH_BINARY, block_size_var.get(), c_var.get())
            self.push_history()
            self.image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            self.gray_image = binary
            self.display_image()
            dialog.destroy()

        # Połączenie zmiennych z funkcją aktualizującą podgląd
        for var in [method_var, block_size_var, c_var]:
            var.trace('w', update_preview)
        update_preview()

        # Przycisk do zatwierdzenia wyboru
        tk.Button(dialog, text="Zastosuj", command=apply).pack(pady=10)

        # Ustawienie rozmiaru okna dialogowego
        dialog.update_idletasks()
        dialog.geometry(f"{dialog.winfo_reqwidth()}x{dialog.winfo_reqheight()}")

    def otsu_threshold(self):
        """
        3.6.1. Segmentacja progowa metodą Otsu.
        """
        # Sprawdzenie, czy obraz jest wczytany
        if self.image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")  # Komunikat, jeśli brak obrazu
            return

        # Konwersja do skali szarości, jeśli potrzeba
        if self.gray_image is None:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # Konwersja do szarości

        self.push_history()
        # Progowanie metodą Otsu (automatyczny dobór progu)
        _, binary = cv2.threshold(self.gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self.gray_image = binary
        self.display_image()

    def grabcut_segmentation(self):
        """
        3.6.2. Segmentacja obrazu metodą GrabCut (interaktywna).
        """
        # Sprawdzenie, czy obraz jest wczytany
        if self.image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")  # Komunikat, jeśli brak obrazu
            return

        self.grabcut_rect = None  # Zmienna na prostokąt zaznaczenia
        self.grabcut_drawing = False  # Flaga rysowania prostokąta

        def start_rect(event):
            # Rozpoczęcie rysowania prostokąta (pobranie pierwszego punktu)
            x = int((event.x - self.image_position[0]) / self.zoom_factor)
            y = int((event.y - self.image_position[1]) / self.zoom_factor)
            self.grabcut_rect = [(x, y)]
            self.grabcut_drawing = True

        def draw_rect(event):
            # Dynamiczne rysowanie prostokąta podczas przeciągania myszy
            if not self.grabcut_drawing:
                return
            x = int((event.x - self.image_position[0]) / self.zoom_factor)
            y = int((event.y - self.image_position[1]) / self.zoom_factor)
            img_copy = self.image.copy()
            cv2.rectangle(img_copy, 
                          self.grabcut_rect[0],
                          (x, y),
                          (0, 255, 0), 2)  # Zielony prostokąt
            self.display_temp_image(img_copy)

        def end_rect(event):
            # Zakończenie rysowania prostokąta (pobranie drugiego punktu)
            if not self.grabcut_drawing:
                return
            x = int((event.x - self.image_position[0]) / self.zoom_factor)
            y = int((event.y - self.image_position[1]) / self.zoom_factor)
            self.grabcut_rect.append((x, y))
            self.grabcut_drawing = False
            self.apply_grabcut()  # Wywołanie segmentacji GrabCut

        # Bindowanie zdarzeń myszy do obsługi rysowania prostokąta
        self.img_panel_container.bind("<Button-1>", start_rect)
        self.img_panel_container.bind("<B1-Motion>", draw_rect)
        self.img_panel_container.bind("<ButtonRelease-1>", end_rect)

        messagebox.showinfo("GrabCut", "Zaznacz prostokąt wokół obiektu")  # Instrukcja dla użytkownika

    def apply_grabcut(self):
        """
        3.6.2. Wykonuje segmentację GrabCut na zaznaczonym prostokącie.
        """
        # Sprawdzenie, czy zaznaczono dwa punkty prostokąta
        if len(self.grabcut_rect) != 2:
            return

        # Przeskaluj współrzędne do rzeczywistego rozmiaru obrazu
        x1 = int(self.grabcut_rect[0][0] / self.zoom_factor)
        y1 = int(self.grabcut_rect[0][1] / self.zoom_factor)
        x2 = int(self.grabcut_rect[1][0] / self.zoom_factor)
        y2 = int(self.grabcut_rect[1][1] / self.zoom_factor)

        # Upewnij się, że współrzędne są w granicach obrazu
        h, w = self.image.shape[:2]
        x1, x2 = max(0, min(x1, w-1)), max(0, min(x2, w-1))
        y1, y2 = max(0, min(y1, h-1)), max(0, min(y2, h-1))

        rect = (min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1))  # Prostokąt do GrabCut

        # Przygotowanie maski i modeli tła/przedmiotu
        mask = np.zeros(self.image.shape[:2], np.uint8)
        bgd_model = np.zeros((1,65), np.float64)
        fgd_model = np.zeros((1,65), np.float64)

        # Właściwa segmentacja GrabCut
        cv2.grabCut(self.image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        # Utworzenie maski binarnej (0/1) na podstawie wyniku GrabCut
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        self.push_history()  # Zapisz stan do historii
        self.image = self.image * mask2[:,:,np.newaxis]  # Zastosuj maskę do obrazu
        self.display_image()  # Odśwież wyświetlany obraz

        # Usuń bindingi po zakończeniu segmentacji
        self.img_panel_container.unbind("<Button-1>")
        self.img_panel_container.unbind("<B1-Motion>")
        self.img_panel_container.unbind("<ButtonRelease-1>")

    def display_temp_image(self, img):
        """
        3.6.2. Tymczasowo wyświetla obraz (np. podgląd segmentacji, maski).
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        new_size = (int(w * self.zoom_factor), int(h * self.zoom_factor))
        img_resized = cv2.resize(img_rgb, new_size)
        self._image_reference_temp = ImageTk.PhotoImage(Image.fromarray(img_resized))
        
        # Rysuj na tym samym płótnie
        if hasattr(self, 'temp_image_id') and self.temp_image_id:
            self.img_panel_container.delete(self.temp_image_id)

        self.temp_image_id = self.img_panel_container.create_image(
            self.image_position[0], self.image_position[1], 
            anchor="nw", image=self._image_reference_temp
        )

    def watershed_segmentation(self):
        """
        3.6.3. Segmentacja obrazu metodą Watershed.
        """
        # Sprawdzenie, czy obraz jest wczytany
        if self.image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")  # Komunikat, jeśli brak obrazu
            return

        # Konwersja na szarość
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Progowanie Otsu (odwrócone, aby tło było białe)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Operacje morfologiczne (otwarcie, aby usunąć szum)
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Pewne tło przez dylatację
        sure_bg = cv2.dilate(opening, kernel, iterations=1)

        # Pewny pierwszy plan przez transformację odległości i progowanie
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.15*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Wyznaczenie nieznanego regionu (tło - pierwszy plan)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Etykietowanie markerów (każdy obiekt dostaje inną etykietę)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1  # Zwiększ etykiety, aby tło miało wartość 1
        markers[unknown == 255] = 0  # Nieznane regiony mają wartość 0

        # Segmentacja metodą Watershed
        markers = cv2.watershed(self.image, markers)

        self.push_history()
        result = self.image.copy()
        self.image = result

        # Obwiedź wszystkie obiekty (poza ramką)
        obiekty = np.zeros_like(markers, dtype=np.uint8)
        obiekty[markers > 1] = 255
        contours, _ = cv2.findContours(obiekty, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        h, w = obiekty.shape
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            if x == 0 and y == 0 and cw == w and ch == h:
                continue  # pomiń ramkę
            cv2.drawContours(self.image, [contour], -1, (0, 255, 0), 1)

        self.display_image()

    def inpainting(self):
        """
        3.6.4. Interaktywna naprawa obrazu (inpainting) na zaznaczonych obszarach.
        """
        # Sprawdzenie, czy obraz jest wczytany
        if self.image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")  # Komunikat, jeśli brak obrazu
            return

        self.inpainting_drawing = False  # Flaga rysowania maski
        self.inpainting_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)  # Maska do naprawy
        self.last_x = None
        self.last_y = None

        def start_drawing(event):
            # Rozpoczęcie rysowania maski (zapamiętaj punkt startowy)
            self.inpainting_drawing = True
            self.last_x = int((event.x - self.image_position[0]) / self.zoom_factor)
            self.last_y = int((event.y - self.image_position[1]) / self.zoom_factor)

        def draw(event):
            # Rysowanie linii maski podczas przeciągania myszy
            if not self.inpainting_drawing:
                return
            x = int((event.x - self.image_position[0]) / self.zoom_factor)
            y = int((event.y - self.image_position[1]) / self.zoom_factor)
            if self.last_x is not None and self.last_y is not None:
                cv2.line(self.inpainting_mask, (self.last_x, self.last_y), (x, y), 255, 5)  # Rysuj na masce
                # Pokaż maskę na obrazie (na zielono)
                img_copy = self.image.copy()
                img_copy[self.inpainting_mask > 0] = [0, 255, 0]
                self.display_temp_image(img_copy)
            self.last_x = x
            self.last_y = y

        def end_drawing(event):
            # Zakończenie rysowania i wykonanie inpaintingu
            self.inpainting_drawing = False
            # Wykonaj inpainting na zaznaczonym obszarze
            self.push_history()  # Zapisz stan do historii
            self.image = cv2.inpaint(self.image, self.inpainting_mask, 3, cv2.INPAINT_TELEA)
            self.display_image()  # Odśwież wyświetlany obraz
            # Reset maski po naprawie
            self.inpainting_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)

        # Bindowanie zdarzeń myszy do obsługi rysowania maski
        self.img_panel_container.bind("<Button-1>", start_drawing)
        self.img_panel_container.bind("<B1-Motion>", draw)
        self.img_panel_container.bind("<ButtonRelease-1>", end_drawing)

        messagebox.showinfo("Inpainting", "Zaznacz obszary do naprawy")  # Instrukcja dla użytkownika

    def rle_compression(self):
        """
        3.6.5. Kompresja obrazu metodą RLE, wyświetlenie stopnia kompresji.
        """
        # Sprawdzenie, czy obraz jest wczytany
        if self.image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")  # Komunikat, jeśli brak obrazu
            return
        # Konwersja do skali szarości, jeśli potrzeba
        if self.gray_image is None:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        rle_data = []  # Lista do przechowywania par (wartość, liczność)
        current_value = self.gray_image[0, 0]  # Początkowa wartość piksela
        count = 1  # Licznik powtórzeń
        # Przejście po wszystkich pikselach obrazu (spłaszczona tablica)
        for pixel in self.gray_image.flatten()[1:]:
            if pixel == current_value:
                count += 1  # Zwiększ licznik jeśli ta sama wartość
            else:
                rle_data.append((current_value, count))  # Dodaj parę do listy
                current_value = pixel  # Zmień wartość
                count = 1  # Resetuj licznik
        rle_data.append((current_value, count))  # Dodaj ostatnią parę
        original_size = self.gray_image.size  # Rozmiar oryginalny (liczba pikseli)
        compressed_size = len(rle_data) * 2  # Rozmiar skompresowany (wartość + liczność)
        compression_ratio = (original_size - compressed_size) / original_size * 100  # Stopień kompresji [%]
        self.rle_compressed_data = rle_data  # Zapisz dane RLE do obiektu
        self.rle_shape = self.gray_image.shape  # Zapisz kształt obrazu
        # Wyświetl informację o stopniu kompresji
        messagebox.showinfo("Kompresja RLE", f"Oryginalny rozmiar: {original_size} bajtów\nSkompresowany rozmiar: {compressed_size} bajtów\nStopień kompresji: {compression_ratio:.2f}%")

    def save_compressed(self):
        """
        3.6.5. Zapisuje skompresowany obraz (RLE) do pliku.
        """
        # Sprawdzenie, czy wykonano kompresję RLE
        if not hasattr(self, 'rle_compressed_data'):
            messagebox.showwarning("Błąd", "Najpierw wykonaj kompresję RLE.")  # Komunikat, jeśli brak danych
            return
        # Okno dialogowe do wyboru ścieżki zapisu
        file_path = filedialog.asksaveasfilename(defaultextension=".rle", filetypes=[("RLE files", "*.rle")])
        if file_path:
            # Zapisz dane do pliku za pomocą pickle
            with open(file_path, 'wb') as f:
                pickle.dump({'shape': self.rle_shape, 'data': self.rle_compressed_data}, f)
            # Komunikat o sukcesie
            messagebox.showinfo("Zapisano", f"Skompresowany obraz zapisany jako: {file_path}")

    def load_compressed(self):
        """
        3.6.5. Wczytuje i dekompresuje obraz z pliku RLE.
        """
        # Okno dialogowe do wyboru pliku RLE
        file_path = filedialog.askopenfilename(filetypes=[("RLE files", "*.rle")])
        if not file_path:
            return
        # Wczytaj dane z pliku pickle
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
            shape = obj['shape']  # Kształt obrazu
            data = obj['data']    # Dane RLE
        decompressed = []  # Lista na zdekompresowane piksele
        # Odtwórz obraz na podstawie par (wartość, liczność)
        for value, count in data:
            decompressed.extend([value] * count)
        self.gray_image = np.array(decompressed, dtype=np.uint8).reshape(shape)  # Odtwórz obraz szarości
        self.image = cv2.cvtColor(self.gray_image, cv2.COLOR_GRAY2BGR)  # Odtwórz obraz kolorowy
        self.display_image()  # Wyświetl obraz
        # Komunikat o sukcesie
        messagebox.showinfo("Wczytano", "Obraz zdekompresowany i wyświetlony.")

    def calculate_moments(self):
        """
        3.6.6. Oblicza i wyświetla momenty obrazu binarnego.
        """
        if self.gray_image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz w skali szarości.")
            return

        # Binaryzacja obrazu
        _, binary = cv2.threshold(self.gray_image, 127, 255, cv2.THRESH_BINARY)
        
        # Oblicz momenty
        moments = cv2.moments(binary)
        
        # Wyświetl wyniki
        result = "Momenty obrazu:\n\n"
        for name, value in moments.items():
            result += f"{name}: {value:.2f}\n"
        
        messagebox.showinfo("Momenty", result)

    def calculate_area_perimeter(self):
        """
        3.6.6. Oblicza i wyświetla pole powierzchni i obwód obiektów na obrazie binarnym.
        """
        if self.gray_image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz w skali szarości.")
            return

        # Binaryzacja obrazu
        _, binary = cv2.threshold(self.gray_image, 127, 255, cv2.THRESH_BINARY)
        
        # Znajdź kontury
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            messagebox.showwarning("Błąd", "Nie znaleziono obiektów na obrazie.")
            return
        
        # Stwórz okno z przewijaniem
        result_window = tk.Toplevel(self.root)
        result_window.title("Pole i obwód")

        # Dodaj frame ze scrollbarem
        main_frame = ttk.Frame(result_window)
        main_frame.pack(fill="both", expand=True)

        # Dodaj Canvas i Scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Rysuj obraz z zaznaczonymi obiektami
        img_with_contours = self.image.copy()
        
        # Oblicz pole i obwód dla każdego konturu
        h, w = binary.shape
        # Oblicz pole i obwód dla każdego konturu
        obj_idx = 1
        for contour in contours:
            # Sprawdź, czy kontur to ramka obrazu
            x, y, cw, ch = cv2.boundingRect(contour)
            if x == 0 and y == 0 and cw == w and ch == h:
                continue  # pomiń ramkę
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            # Rysuj kontur i numer
            cv2.drawContours(img_with_contours, [contour], -1, (0, 255, 0), 2)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(img_with_contours, str(obj_idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # Dodaj informacje o obiekcie
            frame = ttk.LabelFrame(scrollable_frame, text=f"Obiekt {obj_idx}")
            frame.pack(fill="x", padx=5, pady=5)
            ttk.Label(frame, text=f"Pole powierzchni: {area:.2f} pikseli²").pack(anchor="w", padx=5)
            ttk.Label(frame, text=f"Obwód: {perimeter:.2f} pikseli").pack(anchor="w", padx=5)
            obj_idx += 1

        # Wyświetl obraz z zaznaczonymi obiektami
        cv2.imshow("Znalezione obiekty", img_with_contours)

        # Pakowanie elementów
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Dodaj obsługę kółka myszy
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        result_window.update_idletasks()
        result_window.geometry(f"{result_window.winfo_reqwidth()}x{result_window.winfo_reqheight()}")

    def calculate_shape_factors(self):
        """
        3.6.6. Oblicza i wyświetla współczynniki kształtu (aspectRatio, extent, solidity, equivalentDiameter).
        """
        if self.gray_image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz w skali szarości.")
            return

        # Binaryzacja obrazu
        _, binary = cv2.threshold(self.gray_image, 127, 255, cv2.THRESH_BINARY)
        
        # Znajdź kontury
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            messagebox.showwarning("Błąd", "Nie znaleziono obiektów na obrazie.")
            return

        # Stwórz okno z przewijaniem
        result_window = tk.Toplevel(self.root)
        result_window.title("Współczynniki kształtu")

        # Dodaj frame ze scrollbarem
        main_frame = ttk.Frame(result_window)
        main_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Rysuj obraz z zaznaczonymi obiektami
        img_with_contours = self.image.copy()

        for i, contour in enumerate(contours, 1):
            x, y, cw, ch = cv2.boundingRect(contour)
            if x == 0 and y == 0 and cw == binary.shape[1] and ch == binary.shape[0]:
                continue  # pomiń ramkę
            # Podstawowe parametry
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Rysuj kontur i numer
            cv2.drawContours(img_with_contours, [contour], -1, (0, 255, 0), 2)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(img_with_contours, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (255, 0, 0), 2)

            # Oblicz współczynniki kształtu
            if area > 0 and perimeter > 0:
                # Współczynnik kolistości
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Współczynnik zwartości
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                rect_area = cv2.contourArea(box)
                if rect_area > 0:
                    compactness = area / rect_area
                else:
                    compactness = 0
                
                # Współczynnik wydłużenia
                if rect[1][0] > 0 and rect[1][1] > 0:
                    elongation = min(rect[1][0], rect[1][1]) / max(rect[1][0], rect[1][1])
                else:
                    elongation = 0

                # Dodaj informacje o obiekcie
                frame = ttk.LabelFrame(scrollable_frame, text=f"Obiekt {i}")
                frame.pack(fill="x", padx=5, pady=5)
                
                ttk.Label(frame, text=f"Pole powierzchni: {area:.2f} pikseli²").pack(anchor="w", padx=5)
                ttk.Label(frame, text=f"Obwód: {perimeter:.2f} pikseli").pack(anchor="w", padx=5)
                ttk.Label(frame, text=f"Współczynnik kolistości: {circularity:.3f}").pack(anchor="w", padx=5)
                ttk.Label(frame, text=f"Współczynnik zwartości: {compactness:.3f}").pack(anchor="w", padx=5)
                ttk.Label(frame, text=f"Współczynnik wydłużenia: {elongation:.3f}").pack(anchor="w", padx=5)

        # Wyświetl obraz z zaznaczonymi obiektami
        cv2.imshow("Znalezione obiekty", img_with_contours)

        # Pakowanie elementów
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Dodaj obsługę kółka myszy
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        result_window.update_idletasks()
        result_window.geometry(f"{result_window.winfo_reqwidth()}x{result_window.winfo_reqheight()}")

    def calculate_feature_vector(self):
        """
        3.6.6. Oblicza i wyświetla pełny wektor cech obiektów oraz umożliwia eksport do CSV.
        """
        # Sprawdzenie, czy obraz w skali szarości został wczytany
        if self.gray_image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz w skali szarości.")
            return
        # Binarizacja obrazu
        _, binary = cv2.threshold(self.gray_image, 127, 255, cv2.THRESH_BINARY)
        # Wyszukiwanie konturów
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            messagebox.showwarning("Błąd", "Nie znaleziono obiektów na obrazie.")
            return
        # Przygotowanie nagłówków i danych do tabeli oraz CSV
        headers = [
            "Obiekt", "Pole", "Obwód", "Kolistość", "Zwartość", "Wydłużenie", "Prostokątność", "Wypukłość", "Centroid_x", "Centroid_y"
        ]
        rows = []
        img_with_contours = self.image.copy()
        # Obliczanie cech dla każdego konturu
        for i, contour in enumerate(contours, 1):
            x, y, cw, ch = cv2.boundingRect(contour)
            if x == 0 and y == 0 and cw == binary.shape[1] and ch == binary.shape[0]:
                continue  # pomiń ramkę
            area = cv2.contourArea(contour)  # Pole powierzchni
            perimeter = cv2.arcLength(contour, True)  # Obwód
            hull = cv2.convexHull(contour)  # Otoczka wypukła
            hull_area = cv2.contourArea(hull)  # Pole otoczki wypukłej
            rect = cv2.minAreaRect(contour)  # Minimalny prostokąt otaczający
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            rect_area = cv2.contourArea(box)  # Pole prostokąta
            bounding = cv2.boundingRect(contour)  # Prostokąt ograniczający
            bounding_area = bounding[2] * bounding[3]  # Pole prostokąta ograniczającego
            # Kolistość
            if area > 0 and perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            # Zwartość
            if rect_area > 0:
                compactness = area / rect_area
            else:
                compactness = 0
            # Wydłużenie
            if rect[1][0] > 0 and rect[1][1] > 0:
                elongation = min(rect[1][0], rect[1][1]) / max(rect[1][0], rect[1][1])
            else:
                elongation = 0
            # Prostokątność
            if bounding_area > 0:
                rectangularity = area / bounding_area
            else:
                rectangularity = 0
            # Wypukłość
            if hull_area > 0:
                convexity = area / hull_area
            else:
                convexity = 0
            # Wyznaczenie centroidu
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            # Rysowanie konturu i numeru na obrazie
            cv2.drawContours(img_with_contours, [contour], -1, (0, 255, 0), 2)
            cv2.putText(img_with_contours, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # Dodanie cech do wiersza
            rows.append([
                i, area, perimeter, circularity, compactness, elongation, rectangularity, convexity, cx, cy
            ])
        # Okno z tabelą i eksportem
        win = tk.Toplevel(self.root)
        win.title("Wektor cech obiektów")

        # Treeview z przewijaniem
        tree_frame = ttk.Frame(win)
        tree_frame.pack(fill="both", expand=True)

        style = ttk.Style()
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"))
        tree = ttk.Treeview(tree_frame, columns=headers, show="headings", height=15)
        tree.pack(side="left", fill="both", expand=True)

        # Ustawianie nagłówków i szerokości kolumn
        for col in headers:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor="center")

        # Dodawanie danych do tabeli
        for row in rows:
            display_row = [f"{val:.3f}" if isinstance(val, float) else str(val) for val in row]
            tree.insert("", "end", values=display_row)

        # Scrollbar pionowy
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        # Funkcja eksportu do CSV
        def export_csv():
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
            if file_path:
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    for row in rows:
                        writer.writerow(row)
                messagebox.showinfo("Eksport", f"Zapisano do pliku: {file_path}")

        # Przycisk eksportu
        btn = ttk.Button(win, text="Eksportuj do CSV", command=export_csv)
        btn.pack(pady=5)

        # Ustawienie rozmiaru okna
        win.update_idletasks()
        win.geometry(f"{win.winfo_reqwidth()}x{win.winfo_reqheight()}")

    def crop_rectangle_drag(self):
        """
        3.7.1. Kadrowanie obrazu przez przeciągnięcie prostokąta.
        """
        # Sprawdzenie, czy obraz jest wczytany
        if self.image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")
            return

        # Inicjalizacja zmiennych do kadrowania
        self.crop_points = []
        self.cropping = True

        # Zmienne pomocnicze do przechowywania pozycji i ID prostokąta
        start_pos = [0, 0]
        rect_id = [None]  # Użycie listy, aby można było modyfikować wewnątrz funkcji

        def on_press(event):
            # Przelicz współrzędne canvas na współrzędne obrazu i zapisz punkt początkowy
            start_pos[0] = int((event.x - self.image_position[0]) / self.zoom_factor)
            start_pos[1] = int((event.y - self.image_position[1]) / self.zoom_factor)
            
            # Stwórz prostokąt na canvasie przy pierwszym kliknięciu
            if rect_id[0] is None:
                rect_id[0] = self.img_panel_container.create_rectangle(
                    event.x, event.y, event.x, event.y, outline="#00FF00", width=2)

        def on_drag(event):
            # Aktualizuj rozmiar prostokąta podczas przeciągania myszy
            if rect_id[0] is not None:
                start_x_canvas = start_pos[0] * self.zoom_factor + self.image_position[0]
                start_y_canvas = start_pos[1] * self.zoom_factor + self.image_position[1]
                self.img_panel_container.coords(rect_id[0], 
                                              start_x_canvas, 
                                              start_y_canvas, 
                                              event.x, 
                                              event.y)

        def on_release(event):
            # Po puszczeniu przycisku myszy usuń prostokąt z canvasu
            if rect_id[0] is not None:
                self.img_panel_container.delete(rect_id[0])
            
            # Przelicz współrzędne końcowe prostokąta
            end_x = int((event.x - self.image_position[0]) / self.zoom_factor)
            end_y = int((event.y - self.image_position[1]) / self.zoom_factor)
            
            # Uporządkuj współrzędne, aby x1 < x2 i y1 < y2
            x1 = min(start_pos[0], end_x)
            y1 = min(start_pos[1], end_y)
            x2 = max(start_pos[0], end_x)
            y2 = max(start_pos[1], end_y)
            
            # Sprawdź, czy prostokąt jest prawidłowy i otwórz okno dialogowe kadrowania/obrotu
            if x2 > x1 and y2 > y1:
                self.open_crop_rotate_dialog(x1, y1, x2, y2)
            
            # Usuń bindowania zdarzeń myszy i zakończ tryb kadrowania
            self.img_panel_container.unbind("<ButtonPress-1>")
            self.img_panel_container.unbind("<B1-Motion>")
            self.img_panel_container.unbind("<ButtonRelease-1>")
            self.cropping = False

        # Bindowanie zdarzeń myszy do canvasa (rozpoczęcie, przeciąganie, zakończenie)
        self.img_panel_container.bind("<ButtonPress-1>", on_press)
        self.img_panel_container.bind("<B1-Motion>", on_drag)
        self.img_panel_container.bind("<ButtonRelease-1>", on_release)

        # Informacja dla użytkownika o rozpoczęciu kadrowania
        messagebox.showinfo("Kadrowanie", "Narysuj prostokąt na obrazie, aby go wykadrować.")

    def open_crop_rotate_dialog(self, x1, y1, x2, y2):
        # Utwórz okno dialogowe do obrotu i kadrowania
        dialog = tk.Toplevel(self.root)
        dialog.title("Obróć i kadruj")
        dialog.grab_set()

        # Etykieta do podglądu obrazu
        preview_label = tk.Label(dialog)
        preview_label.pack(pady=10)

        # Suwak do wyboru kąta obrotu
        angle_var = tk.DoubleVar(value=0)
        angle_slider = tk.Scale(dialog, from_=-180, to=180, orient=tk.HORIZONTAL,
                                variable=angle_var, length=300, label="Kąt obrotu")
        angle_slider.pack(pady=5)

        rect_coords = (x1, y1, x2, y2)

        def update_preview(*args):
            angle = angle_var.get()
            
            # Kopia bieżącego obrazu do rysowania podglądu
            preview_img = self.image.copy()

            # Pobierz punkty prostokąta
            box = np.array([
                [rect_coords[0], rect_coords[1]],
                [rect_coords[2], rect_coords[1]],
                [rect_coords[2], rect_coords[3]],
                [rect_coords[0], rect_coords[3]]
            ], dtype=np.int32)

            # Wyznacz środek prostokąta
            center_x = (rect_coords[0] + rect_coords[2]) / 2
            center_y = (rect_coords[1] + rect_coords[3]) / 2

            # Macierz rotacji dla wybranego kąta
            rot_mat = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            
            # Obróć punkty prostokąta
            rotated_box = cv2.transform(np.array([box]), rot_mat)[0]

            # Narysuj obrócony prostokąt na podglądzie
            cv2.polylines(preview_img, [rotated_box], isClosed=True, color=(0, 255, 0), thickness=2)

            # Zmień rozmiar do wyświetlenia, ale zachowaj proporcje i ogranicz do rozmiaru ekranu
            max_w = dialog.winfo_screenwidth() * 0.8
            max_h = dialog.winfo_screenheight() * 0.8
            h, w, _ = preview_img.shape

            if h > max_h or w > max_w:
                scale = min(max_h / h, max_w / w)
                new_size = (int(w * scale), int(h * scale))
                preview_img_resized = cv2.resize(preview_img, new_size, interpolation=cv2.INTER_AREA)
            else:
                preview_img_resized = preview_img
            
            # Konwersja do formatu wyświetlanego w Tkinter
            img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(preview_img_resized, cv2.COLOR_BGR2RGB)))
            preview_label.configure(image=img_tk)
            preview_label.image = img_tk

        def order_points(pts):
            # Sortuje punkty do kolejności: lewy górny, prawy górny, prawy dolny, lewy dolny
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]      # lewy górny
            rect[2] = pts[np.argmax(s)]      # prawy dolny
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]   # prawy górny
            rect[3] = pts[np.argmax(diff)]   # lewy dolny
            return rect

        def apply_crop():
            angle = -angle_var.get() # obracanie
            (x1, y1, x2, y2) = rect_coords
            width = x2 - x1
            height = y2 - y1

            # 1. Wyznacz cztery narożniki prostokąta przed obrotem
            box = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], dtype=np.float32)

            # 2. Wyznacz środek prostokąta
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

            # 3. Obróć każdy punkt wokół środka o zadany kąt
            theta = np.deg2rad(angle)
            rot_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            rotated_box = np.dot(box - center, rot_matrix.T) + center

            # 4. Docelowy prostokąt (równe boki)
            dst_rect = np.array([
                [0, 0],
                [width-1, 0],
                [width-1, height-1],
                [0, height-1]
            ], dtype=np.float32)

            # 5. Perspektywiczna transformacja
            M = cv2.getPerspectiveTransform(rotated_box.astype(np.float32), dst_rect)
            warped = cv2.warpPerspective(self.image, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

            self.push_history()
            self.image = warped
            if self.gray_image is not None:
                warped_gray = cv2.warpPerspective(self.gray_image, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                self.gray_image = warped_gray

            self.modified = True
            self.display_image()
            dialog.destroy()

        # Aktualizuj podgląd przy każdej zmianie kąta
        angle_var.trace_add('write', update_preview)
        update_preview()

        # Przyciski do zatwierdzenia lub anulowania operacji
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Zastosuj", command=apply_crop).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Anuluj", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def rotate_90_right(self):
        """
        3.7.2. Obrót obrazu o 90 stopni w prawo.
        """
        # Sprawdzenie, czy obraz został wczytany
        if self.image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")
            return
        # Obrót obrazu kolorowego o 90 stopni w prawo
        self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
        # Obrót obrazu w skali szarości, jeśli istnieje
        if self.gray_image is not None:
            self.gray_image = cv2.rotate(self.gray_image, cv2.ROTATE_90_CLOCKWISE)

        self.push_history()
        self.display_image()

    def rotate_90_left(self):
        """
        3.7.2. Obrót obrazu o 90 stopni w lewo.
        """
        # Sprawdzenie, czy obraz został wczytany
        if self.image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")
            return
        # Obrót obrazu kolorowego o 90 stopni w lewo
        self.image = cv2.rotate(self.image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Obrót obrazu w skali szarości, jeśli istnieje
        if self.gray_image is not None:
            self.gray_image = cv2.rotate(self.gray_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        self.push_history()
        self.display_image()

    def rotate_180(self):
        """
        3.7.3. Obrót obrazu o 180 stopni.
        """
        # Sprawdzenie, czy obraz został wczytany
        if self.image is None:
            messagebox.showwarning("Brak obrazu", "Najpierw wczytaj obraz.")
            return
        # Obrót obrazu kolorowego o 180 stopni
        self.image = cv2.rotate(self.image, cv2.ROTATE_180)
        # Obrót obrazu w skali szarości, jeśli istnieje
        if self.gray_image is not None:
            self.gray_image = cv2.rotate(self.gray_image, cv2.ROTATE_180)

        self.push_history()
        self.display_image()

"""
Główna sekcja uruchamiająca aplikację edytora obrazów
"""
if __name__ == "__main__":
    root = tk.Tk()  # Utworzenie głównego okna aplikacji Tkinter
    app = ImageEditorApp(root)  # Inicjalizacja aplikacji edytora obrazów
    if len(sys.argv) > 1:
        # Jeśli podano argument wiersza poleceń, załaduj projekt z pliku pickle po 100 ms
        root.after(100, lambda: app.load_from_pickle(sys.argv[1]))
    root.mainloop()  # Uruchom główną pętlę aplikacji
