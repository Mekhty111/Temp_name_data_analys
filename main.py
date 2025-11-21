import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import io
import math

ctk.set_appearance_mode("System")  # "Dark", "Light", "System"
ctk.set_default_color_theme("blue")


class RetailSalesApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Retail Sales Analysis")
        self.geometry("1050x700")
        self.resizable(False, False)

        self.chart_visible = False
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

        self.load_btn = ctk.CTkButton(self.sidebar, text="Основная статистика", command=self.dataset_prep)
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

        self.ascii_btn = ctk.CTkButton(self.sidebar, text="ASCII-гистограммы", command=self.ascii_graphic)
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

        # Подфрейм для двух колонок
        self.stats_columns = ctk.CTkFrame(self.stats_frame)
        self.stats_columns.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        mono_font = ("Courier New", 13)  # моноширинный шрифт для ровных ASCII графиков

        # Левая колонка
        self.stats_left = ctk.CTkTextbox(
            self.stats_columns,
            width=500,
            height=250,
            wrap="word",
            font=mono_font,
        )
        self.stats_left.pack(side="left", fill=tk.BOTH, expand=True, padx=(0, 4))

        # Правая колонка
        self.stats_right = ctk.CTkTextbox(
            self.stats_columns,
            width=400,
            height=250,
            wrap="word",
            font=mono_font,
        )
        self.stats_right.pack(side="right", fill=tk.BOTH, expand=False, padx=(4, 0))

        # --- Фрейм с графиками ---
        self.chart_frame = ctk.CTkFrame(self.main_frame, corner_radius=8)

        self.chart_label = ctk.CTkLabel(self.chart_frame, text="Графики", font=("Arial", 14, "bold"))
        self.chart_label.pack(anchor="w", padx=8, pady=4)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    # ---------- Работа с файлом ----------
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

            buffer = io.StringIO()
            self.df.info(buf=buffer)
            info_text = buffer.getvalue()

            full_text = f"Статистика по данным в таблице:\n\n{info_text}"

            self.start_typing(self.stats_left, full_text, delay=5, clear=True)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать CSV:\n{e}")

    # ---------- Анимация печати ----------
    def type_text(self, widget: ctk.CTkTextbox, text: str,
                  i: int = 0, delay: int = 5, clear: bool = False):
        if i == 0:
            widget.configure(state="normal")
            if clear:
                widget.delete("1.0", "end")

        if i < len(text):
            widget.insert("end", text[i])
            widget.see("end")
            self.update_idletasks()
            self.after(delay, self.type_text, widget, text, i + 1, delay, False)
        else:
            widget.configure(state="disabled")

    def start_typing(self, widget: ctk.CTkTextbox, text: str,
                     delay: int = 5, clear: bool = False):
        self.type_text(widget, text, i=0, delay=delay, clear=clear)

    # ---------- Препроцессинг ----------
    def dataset_prep(self):
        if self.df is None:
            msg = "Сначала выберите CSV файл."
            self.start_typing(self.stats_right, msg, delay=5, clear=True)
            return

        # простейший препроцессинг
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df["Quantity"] = self.df["Quantity"].astype("int8")
        self.df["Month"] = self.df["Date"].dt.month_name()

        current = self.stats_right.get("1.0", "end").strip()
        if current == "" or current == "Сначала выберите CSV файл.":
            buffer = io.StringIO()
            self.df.info(buf=buffer)
            info_after = buffer.getvalue()
            msg = (
                "Статистика по данным в таблице после препроцессинга:\n\n"
                "Предобработан признак даты (переведён в datetime), "
                "экстрактированы месяцы из признака даты.\n\n"
                + info_after
            )
            self.start_typing(self.stats_right, msg, delay=5, clear=True)

    # ---------- ASCII-гистограммы ----------
    def ascii_graphic(self):
        def ascii_bar_from_series(
            s: pd.Series,
            label_width: int | None = None,
            bar_width: int = 20,
            char: str = "█",
        ) -> str:
            # (label, value)
            stats = [(str(idx), float(val)) for idx, val in s.items()
                     if pd.api.types.is_number(val)]
            if not stats:
                return "No numeric stats"

            if label_width is None:
                label_width = max(len(name) for name, _ in stats)

            max_val = max(val for _, val in stats) or 1.0

            lines: list[str] = []
            for name, val in stats:
                bar_len = int(val / max_val * bar_width)
                bar = char * bar_len
                line = f"{name:<{label_width}}: {bar:<{bar_width}} {val:.4g}"
                lines.append(line)

            return "\n".join(lines)

        if self.df is None:
            return

        # ЛЕВАЯ ASCII-диаграмма: средняя цена по месяцам
        aggr_month = (
            self.df.groupby("Month")["Price per Unit"]
            .mean()
            .sort_values(ascending=False)
        )

        # ПРАВАЯ ASCII-диаграмма: средняя цена по категориям
        aggr_cat = (
            self.df.groupby("Product Category")["Price per Unit"]
            .mean()
            .sort_values(ascending=False)
        )

        left_label_w = max(len(str(idx)) for idx in aggr_month.index) if len(aggr_month) else 10
        right_label_w = max(len(str(idx)) for idx in aggr_cat.index) if len(aggr_cat) else 10

        msg_left = ascii_bar_from_series(
            aggr_month, label_width=left_label_w, bar_width=22
        )
        self.start_typing(self.stats_left, msg_left, delay=5, clear=True)

        msg_right = ascii_bar_from_series(
            aggr_cat, label_width=right_label_w, bar_width=22
        )
        self.start_typing(self.stats_right, msg_right, delay=5, clear=True)

    # ---------- Показ / скрытие графика ----------
    def toggle_chart(self):
        if not self.chart_visible:
            self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(4, 8))

            self.stats_left.configure(height=120)
            self.stats_right.configure(height=120)
            self.stats_frame.pack_configure(fill=tk.X, expand=False)

            self.chart_btn.configure(text="Скрыть графики")
            self.chart_visible = True

            self.draw_example_chart()
        else:
            self.chart_frame.pack_forget()

            self.stats_left.configure(height=250)
            self.stats_right.configure(height=250)
            self.stats_frame.pack_configure(fill=tk.BOTH, expand=True)

            self.chart_btn.configure(text="Показать графики")
            self.chart_visible = False

    def draw_example_chart(self):
        self.ax.clear()
        if self.df is not None:
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
