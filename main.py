import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import sys
import os

import pandas as pd
import numpy as np
import io
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


PRIMARY = ("#3B82F6", "#2563EB")
ACCENT  = ("#2DD4BF", "#14B8A6")
DANGER  = ("#F97373", "#EF4444")
BG      = "#020617"
CARD_BG = ("#0F172A", "#020617")


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


def normalize_column_name(df, possible_names):
    """Find column name in DataFrame (case-insensitive, handles both Title Case and snake_case)"""
    df_cols = df.columns.tolist()
    for name in possible_names:
        if name in df_cols:
            return name
        for col in df_cols:
            if col.lower() == name.lower():
                return col
    return None


def get_column_names(df):
    """Get normalized column names from DataFrame"""
    return {
        'date': normalize_column_name(df, ['Date', 'date']),
        'category': normalize_column_name(df, ['Product Category', 'product_category', 'ProductCategory']),
        'quantity': normalize_column_name(df, ['Quantity', 'quantity']),
        'price': normalize_column_name(df, ['Price per Unit', 'price_per_unit', 'PricePerUnit']),
        'total': normalize_column_name(df, ['Total Amount', 'total_amount', 'TotalAmount']),
    }


class RetailSalesApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Retail Sales Analysis")
        self.geometry("1040x640")      # a bit lower and narrower
        self.resizable(False, False)
        self.configure(fg_color=BG)

        self.chart_visible = False
        self.df = None
        self.csv_path = None

        self.default_cat_values = ["All categories"]
        self.default_month_values = ["All months"]

        base_font = ("Arial", 12)
        title_font = ("Arial", 20, "bold")
        mono_font = ("Courier New", 11)

        # ---------- Top panel ----------
        self.top_frame = ctk.CTkFrame(self, height=48, corner_radius=12, fg_color=CARD_BG)
        self.top_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        self.title_label = ctk.CTkLabel(
            self.top_frame,
            text="Sales Analysis",
            font=title_font
        )
        self.title_label.pack(side=tk.LEFT, padx=14, pady=6)

        self.file_btn = ctk.CTkButton(
            self.top_frame,
            text="Select CSV",
            command=self.open_file,
            width=120,
            height=26,
            font=base_font,
            fg_color=PRIMARY,
            hover_color="#1D4ED8"
        )
        self.file_btn.pack(side=tk.RIGHT, padx=14, pady=6)

        # ---------- Left sidebar ----------
        self.sidebar = ctk.CTkFrame(self, width=210, corner_radius=14, fg_color=CARD_BG)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 4), pady=4)

        self.menu_label = ctk.CTkLabel(
            self.sidebar,
            text="Menu",
            font=("Arial", 14, "bold")
        )
        self.menu_label.pack(pady=(10, 6))

        btn_kwargs = dict(font=base_font, height=26, fg_color=PRIMARY, hover_color="#1D4ED8")

        self.load_btn = ctk.CTkButton(
            self.sidebar,
            text="Preprocessing",
            command=self.dataset_prep,
            **btn_kwargs
        )
        self.load_btn.pack(fill=tk.X, padx=8, pady=2)

        self.summary_btn = ctk.CTkButton(
            self.sidebar,
            text="Descriptive statistics",
            command=self.statistic_prep,
            **btn_kwargs
        )
        self.summary_btn.pack(fill=tk.X, padx=8, pady=2)

        self.category_btn = ctk.CTkButton(
            self.sidebar,
            text="Sales by category",
            command=self.category_statist,
            **btn_kwargs
        )
        self.category_btn.pack(fill=tk.X, padx=8, pady=2)

        self.month_btn = ctk.CTkButton(
            self.sidebar,
            text="Sales by month",
            command=self.month_statist,
            **btn_kwargs
        )
        self.month_btn.pack(fill=tk.X, padx=8, pady=2)

        self.chart_btn = ctk.CTkButton(
            self.sidebar,
            text="Show charts",
            command=self.toggle_chart,
            **btn_kwargs
        )
        self.chart_btn.pack(fill=tk.X, padx=8, pady=3)

        self.ascii_btn = ctk.CTkButton(
            self.sidebar,
            text="ASCII histograms",
            command=self.ascii_graphic,
            **btn_kwargs
        )
        self.ascii_btn.pack(fill=tk.X, padx=8, pady=2)

        self.cli_btn = ctk.CTkButton(
            self.sidebar,
            text="Open CLI interface",
            command=self.open_cli,
            font=base_font,
            height=26,
            fg_color=ACCENT,
            hover_color="#0D9488",
            text_color="black"
        )
        self.cli_btn.pack(fill=tk.X, padx=8, pady=2)

        # Separator
        self.separator = ctk.CTkFrame(self.sidebar, height=2, fg_color="#1E293B")
        self.separator.pack(fill=tk.X, padx=8, pady=(10, 6))

        # ---------- Parameters ----------
        self.param_label = ctk.CTkLabel(
            self.sidebar,
            text="Parameters",
            font=("Arial", 13, "bold")
        )
        self.param_label.pack(pady=(2, 4))

        self.category_option = ctk.CTkOptionMenu(
            self.sidebar,
            values=self.default_cat_values,
            width=180,
            font=base_font,
            fg_color="#111827",
            button_color="#1F2937",
            button_hover_color="#374151",
            command=self.on_filter_change
        )
        self.category_option.pack(fill=tk.X, padx=10, pady=2)
        self.category_option.set("All categories")

        self.month_option = ctk.CTkOptionMenu(
            self.sidebar,
            values=self.default_month_values,
            width=180,
            font=base_font,
            fg_color="#111827",
            button_color="#1F2937",
            button_hover_color="#374151",
            command=self.on_filter_change
        )
        self.month_option.pack(fill=tk.X, padx=10, pady=2)
        self.month_option.set("All months")

        self.top_entry = ctk.CTkEntry(
            self.sidebar,
            placeholder_text="Top-N categories/months",
            font=base_font,
            height=26,
            fg_color="#020617"
        )
        self.top_entry.pack(fill=tk.X, padx=10, pady=4)

        self.reset_btn = ctk.CTkButton(
            self.sidebar,
            text="Reset selection",
            command=self.reset_filters,
            font=base_font,
            height=26,
            fg_color=ACCENT,
            hover_color="#0D9488",
            text_color="black"
        )
        self.reset_btn.pack(fill=tk.X, padx=8, pady=(2, 6))

        # Spacer
        self.sidebar_spacer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.sidebar_spacer.pack(expand=True, fill=tk.BOTH)

        # Exit button (now very compact)
        self.exit_btn = ctk.CTkButton(
            self.sidebar,
            text="Exit",
            fg_color=DANGER,
            hover_color="#B91C1C",
            command=self.destroy,
            font=base_font,
            height=24
        )
        self.exit_btn.pack(fill=tk.X, padx=8, pady=(2, 8))

        # ---------- Main area ----------
        self.main_frame = ctk.CTkFrame(self, width=800, height=640, corner_radius=14, fg_color=CARD_BG)
        self.main_frame.pack(
            side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 8), pady=4
        )

        self.stats_frame = ctk.CTkFrame(self.main_frame, corner_radius=10, fg_color="#020617")
        self.stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(8, 4))

        self.stats_label = ctk.CTkLabel(
            self.stats_frame,
            text="Key statistics",
            font=("Arial", 13, "bold")
        )
        self.stats_label.pack(anchor="w", padx=8, pady=(6, 2))

        self.stats_columns = ctk.CTkFrame(self.stats_frame, fg_color="transparent")
        self.stats_columns.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 6))

        self.stats_left = ctk.CTkTextbox(
            self.stats_columns,
            width=500,
            height=230,
            wrap="word",
            font=mono_font,
        )
        self.stats_left.pack(side="left", fill=tk.BOTH, expand=True, padx=(0, 4))

        self.stats_right = ctk.CTkTextbox(
            self.stats_columns,
            width=380,
            height=230,
            wrap="word",
            font=mono_font,
        )
        self.stats_right.pack(side="right", fill=tk.BOTH, expand=True, padx=(4, 0))

        # ---------- Charts ----------
        self.chart_frame = ctk.CTkFrame(self.main_frame, corner_radius=10, fg_color="#020617")

        self.chart_label = ctk.CTkLabel(
            self.chart_frame,
            text="Sales charts",
            font=("Arial", 13, "bold")
        )
        self.chart_label.pack(anchor="w", padx=8, pady=(6, 2))

        self.fig = Figure(figsize=(7.0, 2.8), dpi=100)
        self.ax_month = self.fig.add_subplot(1, 2, 1)
        self.ax_cat = self.fig.add_subplot(1, 2, 2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))

    # ----------------- FILTER LOGIC -----------------
    def reset_filters(self):
        self.category_option.set("All categories")
        self.month_option.set("All months")
        self.top_entry.delete(0, "end")
        self.on_filter_change(None)

    def get_column_names(self):
        """Get normalized column names from DataFrame"""
        if self.df is None:
            return None
        return get_column_names(self.df)

    def get_filtered_df(self):
        if self.df is None:
            return None

        df = self.df.copy()
        cat_val = self.category_option.get()
        month_val = self.month_option.get()
        
        cols = get_column_names(df)
        cat_col = cols.get('category') or "Product Category"
        month_col = "Month"

        if cat_col in df.columns and cat_val and cat_val != "All categories":
            df = df[df[cat_col] == cat_val]

        if month_col in df.columns and month_val and month_val != "All months":
            df = df[df[month_col] == month_val]

        return df

    def on_filter_change(self, _value):
        df = self.get_filtered_df()
        if df is None or df.empty:
            self.start_typing(
                self.stats_left,
                "No data for current filters.",
                delay=2,
                clear=True
            )
            if self.chart_visible:
                self.draw_matplotlib_charts()
            return

        cols = get_column_names(df)
        lines = ["Statistics for selected filters:\n"]
        
        total_col = cols.get('total') or "Total Amount"
        qty_col = cols.get('quantity') or "Quantity"
        price_col = cols.get('price') or "Price per Unit"
        
        if total_col in df.columns:
            total_series = df[total_col]
            lines.append(f"Total Amount:")
            lines.append(f"  Count:  {total_series.count()}")
            lines.append(f"  Sum:    {total_series.sum():.2f}")
            lines.append(f"  Mean:   {total_series.mean():.2f}")
            lines.append(f"  Median: {total_series.median():.2f}")
            lines.append(f"  Std:    {total_series.std():.2f}")
            lines.append(f"  Min:    {total_series.min():.2f}")
            lines.append(f"  Max:    {total_series.max():.2f}")
        
        if qty_col in df.columns:
            qty_series = df[qty_col]
            lines.append(f"\nQuantity:")
            lines.append(f"  Count:  {qty_series.count()}")
            lines.append(f"  Sum:    {qty_series.sum():.0f}")
            lines.append(f"  Mean:   {qty_series.mean():.2f}")
            lines.append(f"  Median: {qty_series.median():.2f}")
            lines.append(f"  Std:    {qty_series.std():.2f}")
            lines.append(f"  Min:    {qty_series.min():.0f}")
            lines.append(f"  Max:    {qty_series.max():.0f}")

        self.start_typing(
            self.stats_left,
            "\n".join(lines),
            delay=2,
            clear=True
        )

        if self.chart_visible:
            self.draw_matplotlib_charts()

    # ----------------- FILE / TEXT -----------------
    def open_file(self):
        path = filedialog.askopenfilename(
            title="Select CSV file",
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

            full_text = f"Dataset statistics:\n\n{info_text}"
            self.start_typing(self.stats_left, full_text, delay=2, clear=True)

            cols = get_column_names(self.df)
            cat_col = cols.get('category') or "Product Category"
            
            if cat_col in self.df.columns:
                cats = sorted(self.df[cat_col].dropna().unique().tolist())
                self.category_option.configure(values=["All categories"] + cats)
                self.category_option.set("All categories")

            date_col = cols.get('date') or "Date"
            if date_col in self.df.columns:
                try:
                    self.df[date_col] = pd.to_datetime(self.df[date_col])
                    self.df["Month"] = self.df[date_col].dt.month_name()
                    months = self.df["Month"].dropna().unique().tolist()
                    month_order = [
                        "January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December"
                    ]
                    ordered = [m for m in month_order if m in months]
                    self.month_option.configure(values=["All months"] + ordered)
                    self.month_option.set("All months")
                except Exception:
                    pass

        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV:\n{e}")

    # make textbox read-only (state="disabled" after writing)
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
            widget.configure(state="disabled")  # disable manual input

    def start_typing(self, widget: ctk.CTkTextbox, text: str,
                     delay: int = 5, clear: bool = False):
        self.type_text(widget, text, i=0, delay=delay, clear=clear)

    # ----------------- PREPROCESSING + CLEANING -----------------
    def dataset_prep(self):
        """
        Preprocess dataset:
        - Parse dates
        - Ensure types for key columns
        - Drop missing values in key columns
        - Detect and remove outliers in numeric columns (IQR method)
        - Show cleaning summary in LEFT textbox
        - Show resulting DataFrame info in RIGHT textbox
        """
        if self.df is None:
            msg = "Please select a CSV file first."
            self.start_typing(self.stats_right, msg, delay=5, clear=True)
            return

        df = self.df.copy()
        original_rows = len(df)

        # Get column names (normalized)
        cols = get_column_names(df)
        date_col = cols.get('date') or "Date"
        cat_col = cols.get('category') or "Product Category"
        qty_col = cols.get('quantity') or "Quantity"
        price_col = cols.get('price') or "Price per Unit"
        total_col = cols.get('total') or "Total Amount"

        summary_lines = []
        summary_lines.append("Preprocessing and data cleaning report:\n")
        summary_lines.append(f"Initial number of rows: {original_rows}")

        # --- Date parsing ---
        if date_col in df.columns:
            before_invalid = df[date_col].isna().sum()
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            after_invalid = df[date_col].isna().sum()
            summary_lines.append(
                f"- {date_col}: converted to datetime, "
                f"coerced to NaT: {after_invalid - before_invalid}"
            )
        else:
            summary_lines.append(f"- Column '{date_col}' not found.")

        # --- Ensure numeric columns ---
        numeric_cols = []
        for col in [qty_col, price_col, total_col]:
            if col in df.columns:
                before_non_numeric = df[col].isna().sum()
                df[col] = pd.to_numeric(df[col], errors="coerce")
                after_non_numeric = df[col].isna().sum()
                summary_lines.append(
                    f"- {col}: converted to numeric, "
                    f"non-numeric to NaN: {after_non_numeric - before_non_numeric}"
                )
                numeric_cols.append(col)
                # Convert quantity to int
                if col == qty_col:
                    df[col] = df[col].fillna(0).astype(int)
                    summary_lines.append(f"- {col}: converted to integer type")
            else:
                summary_lines.append(f"- Column '{col}' not found.")

        # --- Drop rows with missing values in key columns ---
        key_cols = [c for c in [date_col, cat_col, qty_col, price_col, total_col] if c in df.columns]
        if key_cols:
            before_drop_na = len(df)
            df_clean_na = df.dropna(subset=key_cols)
            dropped_na = before_drop_na - len(df_clean_na)
            summary_lines.append(
                f"- Rows dropped due to missing values in {key_cols}: {dropped_na}"
            )
            df = df_clean_na
        else:
            summary_lines.append("- No key columns found to check missing values.")

        # --- Add Month column (after Date is cleaned) ---
        if date_col in df.columns:
            df["Month"] = df[date_col].dt.month_name()
            summary_lines.append("- 'Month' column added from 'Date'.")

        # --- Detect and remove outliers (IQR) in numeric columns ---
        def iqr_filter(series: pd.Series):
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0 or np.isnan(iqr):
                return pd.Series([False] * len(series), index=series.index)
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return (series < lower) | (series > upper)

        total_outliers = 0
        if numeric_cols:
            outlier_mask = pd.Series(False, index=df.index)
            for col in numeric_cols:
                mask = iqr_filter(df[col].dropna())
                # align mask to df index
                mask = mask.reindex(df.index, fill_value=False)
                n_outliers = mask.sum()
                summary_lines.append(
                    f"- Outliers detected in '{col}' (IQR 1.5×IQR rule): {n_outliers}"
                )
                outlier_mask |= mask

            before_outliers = len(df)
            df_no_outliers = df[~outlier_mask]
            total_outliers = before_outliers - len(df_no_outliers)
            summary_lines.append(
                f"- Total rows removed as outliers (across numeric columns): {total_outliers}"
            )
            df = df_no_outliers
        else:
            summary_lines.append("- No numeric columns found for outlier detection.")

        final_rows = len(df)
        summary_lines.append(f"\nFinal number of rows after cleaning: {final_rows}")
        summary_text = "\n".join(summary_lines)

        # Save back cleaned df
        self.df = df

        # Info after preprocessing
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        info_after = buffer.getvalue()

        right_msg = "DataFrame info after preprocessing:\n\n" + info_after

        # LEFT: cleaning report, RIGHT: final info
        self.start_typing(self.stats_left, summary_text, delay=2, clear=True)
        self.start_typing(self.stats_right, right_msg, delay=2, clear=True)

    def _ascii_bar_from_series(
        self,
        s: pd.Series,
        label_width: int | None = None,
        bar_width: int = 20,
        char: str = "█",
    ) -> str:
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

    def ascii_graphic(self):
        if self.df is None:
            return

        df = self.get_filtered_df()
        if df is None or df.empty:
            return

        cols = get_column_names(df)
        total_col = cols.get('total') or "Total Amount"
        cat_col = cols.get('category') or "Product Category"
        
        # Aggregate by total sales (sum), not average price
        aggr_month = (
            df.groupby("Month")[total_col]
            .sum()
            .sort_values(ascending=False)
        )
        aggr_cat = (
            df.groupby(cat_col)[total_col]
            .sum()
            .sort_values(ascending=False)
        )
        
        n_cat = self.top_entry.get()
        try:
            n_val = int(n_cat)
            if n_val > 0:
                aggr_month = aggr_month.head(n_val)
                aggr_cat = aggr_cat.head(n_val)
        except Exception:
            pass

        left_label_w = max(len(str(idx)) for idx in aggr_month.index) if len(aggr_month) else 10
        right_label_w = max(len(str(idx)) for idx in aggr_cat.index) if len(aggr_cat) else 10

        msg_left = "Sales by Month (Total Amount):\n\n" + self._ascii_bar_from_series(
            aggr_month, label_width=left_label_w, bar_width=22
        )
        msg_right = "Sales by Category (Total Amount):\n\n" + self._ascii_bar_from_series(
            aggr_cat, label_width=right_label_w, bar_width=22
        )

        self.start_typing(self.stats_left, msg_left, delay=2, clear=True)
        self.start_typing(self.stats_right, msg_right, delay=2, clear=True)

    # ----------------- DESCRIPTIVE STATISTICS -----------------
    def statistic_prep(self):
        if self.df is None:
            return
        df = self.get_filtered_df()
        if df is None or df.empty:
            return

        cols = get_column_names(df)
        total_col = cols.get('total') or "Total Amount"
        qty_col = cols.get('quantity') or "Quantity"
        
        lines_left = ["Descriptive Statistics\n"]
        lines_left.append("=" * 50)
        
        if total_col in df.columns:
            total_series = df[total_col]
            lines_left.append(f"\nTotal Amount:")
            lines_left.append(f"  Count:  {total_series.count()}")
            lines_left.append(f"  Mean:   {total_series.mean():.2f}")
            lines_left.append(f"  Median: {total_series.median():.2f}")
            lines_left.append(f"  Std:    {total_series.std():.2f}")
            lines_left.append(f"  Min:    {total_series.min():.2f}")
            lines_left.append(f"  Max:    {total_series.max():.2f}")
        
        lines_right = []
        if qty_col in df.columns:
            qty_series = df[qty_col]
            lines_right.append("Quantity:")
            lines_right.append(f"  Count:  {qty_series.count()}")
            lines_right.append(f"  Mean:   {qty_series.mean():.2f}")
            lines_right.append(f"  Median: {qty_series.median():.2f}")
            lines_right.append(f"  Std:    {qty_series.std():.2f}")
            lines_right.append(f"  Min:    {qty_series.min():.0f}")
            lines_right.append(f"  Max:    {qty_series.max():.0f}")

        self.start_typing(
            self.stats_left,
            "\n".join(lines_left),
            delay=2,
            clear=True
        )
        self.start_typing(
            self.stats_right,
            "\n".join(lines_right),
            delay=2,
            clear=True
        )

    # ----------------- ASCII BY CATEGORY / MONTH -----------------
    def category_statist(self):
        if self.df is None:
            return

        df = self.get_filtered_df()
        if df is None or df.empty:
            return

        cols = get_column_names(df)
        total_col = cols.get('total') or "Total Amount"
        cat_col = cols.get('category') or "Product Category"
        
        # Aggregate by total sales (sum), not average price
        aggr_cat = (
            df.groupby(cat_col)[total_col]
            .sum()
            .sort_values(ascending=False)
        )
        
        n_cat = self.top_entry.get()
        try:
            n_val = int(n_cat)
            if n_val > 0:
                aggr_cat = aggr_cat.head(n_val)
        except Exception:
            pass

        left_label_w = max(len(str(idx)) for idx in aggr_cat.index) if len(aggr_cat) else 10

        msg_left = "Sales by Category (Total Amount):\n\n" + self._ascii_bar_from_series(
            aggr_cat, label_width=left_label_w, bar_width=22
        )
        self.start_typing(self.stats_left, msg_left, delay=2, clear=True)

    def month_statist(self):
        if self.df is None:
            return

        df = self.get_filtered_df()
        if df is None or df.empty:
            return

        cols = get_column_names(df)
        total_col = cols.get('total') or "Total Amount"
        
        # Aggregate by total sales (sum), not average price
        aggr_month = (
            df.groupby("Month")[total_col]
            .sum()
            .sort_values(ascending=False)
        )
        
        # Order months properly
        month_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        aggr_month = aggr_month.reindex(
            [m for m in month_order if m in aggr_month.index]
        )
        
        n_cat = self.top_entry.get()
        try:
            n_val = int(n_cat)
            if n_val > 0:
                aggr_month = aggr_month.head(n_val)
        except Exception:
            pass

        right_label_w = max(len(str(idx)) for idx in aggr_month.index) if len(aggr_month) else 10

        msg_right = "Sales by Month (Total Amount):\n\n" + self._ascii_bar_from_series(
            aggr_month, label_width=right_label_w, bar_width=22
        )
        self.start_typing(self.stats_right, msg_right, delay=2, clear=True)

    # ----------------- CHARTS -----------------
    def toggle_chart(self):
        if not self.chart_visible:
            self.chart_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=(2, 8))

            self.stats_left.configure(height=150)
            self.stats_right.configure(height=150)
            self.stats_frame.pack_configure(fill=tk.X, expand=False)

            self.chart_btn.configure(text="Hide charts")
            self.chart_visible = True

            self.draw_matplotlib_charts()
        else:
            self.chart_frame.pack_forget()

            self.stats_left.configure(height=230)
            self.stats_right.configure(height=230)
            self.stats_frame.pack_configure(fill=tk.BOTH, expand=True)

            self.chart_btn.configure(text="Show charts")
            self.chart_visible = False

    def draw_matplotlib_charts(self):
        self.ax_month.clear()
        self.ax_cat.clear()

        df = self.get_filtered_df()
        if df is None or df.empty:
            self.ax_month.text(
                0.5, 0.5, "No data\n(load CSV or change filters)",
                ha="center", va="center", fontsize=9
            )
            self.ax_month.axis("off")
            self.ax_cat.axis("off")
            self.canvas.draw()
            return

        cols = get_column_names(df)
        total_col = cols.get('total') or "Total Amount"
        cat_col = cols.get('category') or "Product Category"
        
        # Aggregate by total sales (sum), not average price
        monthly_sales = df.groupby("Month")[total_col].sum()
        month_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        monthly_sales = monthly_sales.reindex(
            [m for m in month_order if m in monthly_sales.index]
        )

        category_sales = (
            df.groupby(cat_col)[total_col]
            .sum()
            .sort_values(ascending=False)
        )

        n_cat = self.top_entry.get()
        try:
            n_val = int(n_cat)
            if n_val > 0:
                monthly_sales = monthly_sales.head(n_val)
                category_sales = category_sales.head(n_val)
        except Exception:
            pass

        if len(monthly_sales):
            x = np.arange(len(monthly_sales))
            self.ax_month.bar(x, monthly_sales.values, color="#3B82F6")
            self.ax_month.set_xticks(x)
            self.ax_month.set_xticklabels(
                monthly_sales.index,
                rotation=45,
                ha="right",
                fontsize=8
            )
            self.ax_month.set_title("Total sales by month", fontsize=9)
            self.ax_month.set_ylabel("Total Amount", fontsize=8)
            self.ax_month.grid(axis="y", linestyle="--", alpha=0.3)
        else:
            self.ax_month.text(
                0.5, 0.5, "No month data",
                ha="center", va="center", fontsize=9
            )
            self.ax_month.axis("off")

        if len(category_sales):
            x2 = np.arange(len(category_sales))
            self.ax_cat.bar(x2, category_sales.values, color="#FBBF24")
            self.ax_cat.set_xticks(x2)
            self.ax_cat.set_xticklabels(
                category_sales.index,
                rotation=45,
                ha="right",
                fontsize=8
            )
            self.ax_cat.set_title("Total sales by category", fontsize=9)
            self.ax_cat.set_ylabel("Total Amount", fontsize=8)
            self.ax_cat.grid(axis="y", linestyle="--", alpha=0.3)
        else:
            self.ax_cat.text(
                0.5, 0.5, "No category data",
                ha="center", va="center", fontsize=9
            )
            self.ax_cat.axis("off")

        self.fig.tight_layout()
        self.canvas.draw()

    def open_cli(self):
        """Open CLI interface in a new terminal window"""
        try:
            script_path = os.path.join(os.path.dirname(__file__), "cli_interface.py")
            python_exe = sys.executable
            
            if sys.platform == "win32":
                subprocess.Popen(["start", "cmd", "/k", f"{python_exe} {script_path}"], shell=True)
            elif sys.platform == "darwin":  # macOS
                # Use osascript to open Terminal and run the script
                cmd = f'osascript -e \'tell application "Terminal" to do script "{python_exe} {script_path}"\''
                subprocess.Popen(cmd, shell=True)
            else:  # Linux
                subprocess.Popen(["xterm", "-e", f"{python_exe} {script_path}"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open CLI interface:\n{e}\n\nYou can run it manually:\npython cli_interface.py")


if __name__ == "__main__":
    app = RetailSalesApp()
    app.mainloop()
