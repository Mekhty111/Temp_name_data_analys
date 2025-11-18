import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import io


ctk.set_appearance_mode("System")  # "Dark", "Light", "System"
ctk.set_default_color_theme("blue")


class RetailSalesApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Retail Sales Analysis")
        self.geometry("1050x700")
        self.resizable(False, False)

        self.chart_visible = False      # флаг видимости графика
        self.df = None
        self.csv_path = None

        # --- Верхний фрейм ---
        self.top_frame = ctk.CTkFrame(self, height=50)
        self.top_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        self.title_label = ctk.CTkLabel(
            self.top_frame,
            text="Анализ Продаж",
            font=("Arial", 22, "bold")
        )
        self.title_label.pack(side=tk.LEFT, padx=16)

        self.file_btn = ctk.CTkButton(
            self.top_frame,
            text="Выбрать CSV",
            command=self.open_file
        )
        self.file_btn.pack(side=tk.RIGHT, padx=16)

        # --- Левый сайдбар ---
        self.sidebar = ctk.CTkFrame(self, width=220)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 5), pady=5)

        self.menu_label = ctk.CTkLabel(self.sidebar, text="Меню", font=("Arial", 16, "bold"))
        self.menu_label.pack(pady=(10, 12))

        self.load_btn = ctk.CTkButton(self.sidebar, text="Загрузить данные")
        self.load_btn.pack(fill=tk.X, pady=6)

        self.summary_btn = ctk.CTkButton(self.sidebar, text="Cтатистика")
        self.summary_btn.pack(fill=tk.X, pady=6)

        self.category_btn = ctk.CTkButton(self.sidebar, text="Продажи по категориям")
        self.category_btn.pack(fill=tk.X, pady=6)

        self.month_btn = ctk.CTkButton(self.sidebar, text="Продажи по месяцам")
        self.month_btn.pack(fill=tk.X, pady=6)

        self.chart_btn = ctk.CTkButton(
            self.sidebar,
            text="Показать графики",
            command=self.toggle_chart
        )
        self.chart_btn.pack(fill=tk.X, pady=6)

        self.ascii_btn = ctk.CTkButton(self.sidebar, text="ASCII-гистограммы")
        self.ascii_btn.pack(fill=tk.X, pady=6)

        self.exit_btn = ctk.CTkButton(
            self.sidebar,
            text="Выйти",
            fg_color="#C82333",
            hover_color="#A71D2A",
            command=self.destroy
        )
        self.exit_btn.pack(fill=tk.X, pady=(24, 6))

        # --- Параметры ---
        self.param_label = ctk.CTkLabel(self.sidebar, text="Параметры", font=("Arial", 14, "bold"))
        self.param_label.pack(pady=(28, 2))

        self.category_option = ctk.CTkOptionMenu(self.sidebar, values=["Все категории"])
        self.category_option.pack(fill=tk.X, padx=2, pady=4)

        self.month_option = ctk.CTkOptionMenu(self.sidebar, values=["Все месяцы"])
        self.month_option.pack(fill=tk.X, padx=2, pady=4)

        self.top_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Top-N категорий")
        self.top_entry.pack(fill=tk.X, padx=2, pady=4)

        # --- Центральная зона ---
        self.main_frame = ctk.CTkFrame(self, width=800, height=700)
        self.main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=5)

        # Фрейм статистики
        self.stats_frame = ctk.CTkFrame(self.main_frame, corner_radius=8)
        self.stats_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(10, 8))

        self.stats_label = ctk.CTkLabel(
            self.stats_frame,
            text="Ключевая статистика",
            font=("Arial", 14, "bold")
        )
        self.stats_label.pack(anchor="w", padx=8, pady=4)

        # Изначально большой текстбокс
        self.stats_text = ctk.CTkTextbox(
            self.stats_frame,
            width=800,
            height=250,
            wrap="word"
        )
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Фрейм с графиками (изначально скрыт — без pack)
        self.chart_frame = ctk.CTkFrame(self.main_frame, corner_radius=8)

        self.chart_label = ctk.CTkLabel(self.chart_frame, text="Графики", font=("Arial", 14, "bold"))
        self.chart_label.pack(anchor="w", padx=8, pady=4)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    # ---------- Логика работы с файлом ----------

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Выберите CSV файл",
            filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return

        try:
            self.df = pd.read_csv(path)
            self.csv_path = path

            # Получаем info() в строку
            buffer = io.StringIO()
            self.df.info(buf=buffer)
            info_text = buffer.getvalue()

            full_text = f"Статистика по данным в таблице: \n\n{info_text}"

            self.start_typing(full_text)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать CSV:\n{e}")

    # ---------- Анимация текста ----------

    def type_text(self, text: str, i: int = 0, delay: int = 5):
        """Анимация: печатает text по одному символу в stats_text."""
        if i == 0:
            self.stats_text.configure(state="normal")
            self.stats_text.delete("1.0", "end")

        if i < len(text):
            self.stats_text.insert("end", text[i])
            self.stats_text.see("end")
            self.update_idletasks()
            self.after(delay, self.type_text, text, i + 1, delay)
        else:
            self.stats_text.configure(state="disabled")

    def start_typing(self, text: str, delay: int = 5):
        self.type_text(text, i=0, delay=delay)

    # ---------- Показ / скрытие графика + изменение layout ----------

    def toggle_chart(self):
        if not self.chart_visible:
            # показать график
            self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(4, 8))

            # уменьшить текстбокс, чтобы освободить место графику
            self.stats_text.configure(height=120)
            self.stats_frame.pack_configure(fill=tk.X, expand=False)

            self.chart_btn.configure(text="Скрыть графики")
            self.chart_visible = True

            # Пример: если df есть, нарисовать что‑нибудь простое
            self.draw_example_chart()
        else:
            # спрятать график
            self.chart_frame.pack_forget()

            # вернуть текстбокс побольше
            self.stats_text.configure(height=250)
            self.stats_frame.pack_configure(fill=tk.BOTH, expand=True)

            self.chart_btn.configure(text="Показать графики")
            self.chart_visible = False

    def draw_example_chart(self):
        """Пример: рисуем рандомный график или что‑то из df."""
        self.ax.clear()
        if self.df is not None:
            # если хочешь — тут вставь свою логику графиков по df
            # сейчас просто пример с рандомом
            x = np.arange(10)
            y = np.random.randint(1, 100, size=10)
            self.ax.plot(x, y, marker="o")
            self.ax.set_title("Пример графика")
        else:
            x = np.arange(10)
            y = np.random.randint(1, 100, size=10)
            self.ax.plot(x, y, marker="o")
            self.ax.set_title("Загрузите CSV для реальных данных")

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.fig.tight_layout()
        self.canvas.draw_idle()


if __name__ == "__main__":
    app = RetailSalesApp()
    app.mainloop()
