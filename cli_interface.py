"""
Text-based CLI interface for Sales Data Analysis Application
"""
import pandas as pd
import os
import sys


def normalize_column_name(df, possible_names):
    """Find column name in DataFrame (case-insensitive, handles both Title Case and snake_case)"""
    df_cols = df.columns.tolist()
    for name in possible_names:
        # Exact match
        if name in df_cols:
            return name
        # Case-insensitive match
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


def load_dataset(path):
    """Load and return DataFrame from CSV file"""
    if not os.path.exists(path):
        print(f"Error: File '{path}' not found.")
        return None
    try:
        df = pd.read_csv(path)
        print(f"Dataset loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def clean_data(df):
    """Clean and preprocess the dataset"""
    if df is None:
        return None
    
    cols = get_column_names(df)
    df = df.copy()
    original_rows = len(df)
    
    print("\n=== Data Cleaning ===")
    print(f"Initial rows: {original_rows}")
    
    # Convert date
    if cols['date']:
        df[cols['date']] = pd.to_datetime(df[cols['date']], errors='coerce')
        df['Month'] = df[cols['date']].dt.month_name()
        print(f"✓ Date column converted to datetime")
    
    # Convert quantity to int
    if cols['quantity']:
        df[cols['quantity']] = pd.to_numeric(df[cols['quantity']], errors='coerce')
        df[cols['quantity']] = df[cols['quantity']].fillna(0).astype(int)
        print(f"✓ Quantity converted to integer")
    
    # Convert numeric columns
    for col_name, col_key in [('price', 'price'), ('total', 'total')]:
        if cols[col_key]:
            df[cols[col_key]] = pd.to_numeric(df[cols[col_key]], errors='coerce')
            print(f"✓ {cols[col_key]} converted to numeric")
    
    # Drop rows with missing values in key columns
    key_cols = [c for c in [cols['date'], cols['category'], cols['quantity'], 
                            cols['price'], cols['total']] if c]
    if key_cols:
        before = len(df)
        df = df.dropna(subset=key_cols)
        dropped = before - len(df)
        print(f"✓ Dropped {dropped} rows with missing values")
    
    print(f"Final rows: {len(df)}")
    return df


def show_summary_statistics(df):
    """Show descriptive statistics for Total Amount and Quantity"""
    if df is None:
        print("Error: No dataset loaded. Please load a dataset first.")
        return
    
    cols = get_column_names(df)
    
    print("\n=== Summary Statistics ===")
    
    if cols['total']:
        total_col = df[cols['total']]
        print(f"\nTotal Amount:")
        print(f"  Mean:   {total_col.mean():.2f}")
        print(f"  Median: {total_col.median():.2f}")
        print(f"  Std:    {total_col.std():.2f}")
        print(f"  Min:    {total_col.min():.2f}")
        print(f"  Max:    {total_col.max():.2f}")
        print(f"  Count:  {total_col.count()}")
    
    if cols['quantity']:
        qty_col = df[cols['quantity']]
        print(f"\nQuantity:")
        print(f"  Mean:   {qty_col.mean():.2f}")
        print(f"  Median: {qty_col.median():.2f}")
        print(f"  Std:    {qty_col.std():.2f}")
        print(f"  Min:    {qty_col.min():.0f}")
        print(f"  Max:    {qty_col.max():.0f}")
        print(f"  Count:  {qty_col.count()}")


def show_sales_by_category(df, top_n=None):
    """Show sales aggregated by category"""
    if df is None:
        print("Error: No dataset loaded. Please load a dataset first.")
        return
    
    cols = get_column_names(df)
    
    if not cols['category'] or not cols['total']:
        print("Error: Required columns not found.")
        return
    
    print("\n=== Sales by Category ===")
    
    sales_by_cat = df.groupby(cols['category'])[cols['total']].sum().sort_values(ascending=False)
    
    if top_n:
        try:
            top_n = int(top_n)
            sales_by_cat = sales_by_cat.head(top_n)
            print(f"Top {top_n} categories:\n")
        except ValueError:
            pass
    
    print(f"{'Category':<20} {'Total Sales':>15}")
    print("-" * 36)
    for cat, sales in sales_by_cat.items():
        print(f"{str(cat):<20} {sales:>15.2f}")


def show_sales_by_month(df, top_n=None):
    """Show sales aggregated by month"""
    if df is None:
        print("Error: No dataset loaded. Please load a dataset first.")
        return
    
    cols = get_column_names(df)
    
    if not cols['total'] or 'Month' not in df.columns:
        print("Error: Required columns not found.")
        return
    
    print("\n=== Sales by Month ===")
    
    sales_by_month = df.groupby('Month')[cols['total']].sum()
    
    # Order months
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    sales_by_month = sales_by_month.reindex(
        [m for m in month_order if m in sales_by_month.index]
    )
    
    if top_n:
        try:
            top_n = int(top_n)
            sales_by_month = sales_by_month.head(top_n)
            print(f"Top {top_n} months:\n")
        except ValueError:
            pass
    
    print(f"{'Month':<15} {'Total Sales':>15}")
    print("-" * 31)
    for month, sales in sales_by_month.items():
        print(f"{str(month):<15} {sales:>15.2f}")


def create_ascii_histogram(series, title="", bar_width=30):
    """Create ASCII histogram from a pandas Series"""
    if series.empty:
        print("No data to display")
        return
    
    print(f"\n=== {title} ===")
    
    max_val = series.max()
    if max_val == 0:
        print("All values are zero")
        return
    
    max_label_len = max(len(str(idx)) for idx in series.index)
    
    for idx, val in series.items():
        bar_len = int((val / max_val) * bar_width)
        bar = "█" * bar_len
        print(f"{str(idx):<{max_label_len}} │{bar} {val:.2f}")


def show_ascii_histograms(df, top_n=None):
    """Show ASCII histograms for sales by category and month"""
    if df is None:
        print("Error: No dataset loaded. Please load a dataset first.")
        return
    
    cols = get_column_names(df)
    
    if not cols['category'] or not cols['total']:
        print("Error: Required columns not found.")
        return
    
    # Sales by category
    sales_by_cat = df.groupby(cols['category'])[cols['total']].sum().sort_values(ascending=False)
    if top_n:
        try:
            sales_by_cat = sales_by_cat.head(int(top_n))
        except ValueError:
            pass
    create_ascii_histogram(sales_by_cat, "Sales by Category", bar_width=30)
    
    # Sales by month
    if 'Month' in df.columns:
        sales_by_month = df.groupby('Month')[cols['total']].sum()
        month_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        sales_by_month = sales_by_month.reindex(
            [m for m in month_order if m in sales_by_month.index]
        )
        if top_n:
            try:
                sales_by_month = sales_by_month.head(int(top_n))
            except ValueError:
                pass
        create_ascii_histogram(sales_by_month, "Sales by Month", bar_width=30)


def print_menu():
    """Print the main menu"""
    print("\n" + "="*50)
    print("  Sales Data Analysis Application")
    print("="*50)
    print("1. Load dataset")
    print("2. Show summary statistics")
    print("3. Show sales by category")
    print("4. Show sales by month")
    print("5. Show ASCII histograms")
    print("0. Exit")
    print("="*50)


def main():
    """Main CLI loop"""
    df = None
    
    print("Welcome to Sales Data Analysis Application!")
    print("This is the text-based interface.")
    
    while True:
        print_menu()
        choice = input("\nEnter your choice: ").strip()
        
        if choice == "0":
            print("Goodbye!")
            break
        elif choice == "1":
            path = input("Enter path to CSV file (or press Enter for 'retail_sales_dataset.csv'): ").strip()
            if not path:
                # Try default dataset
                default_path = "retail_sales_dataset.csv"
                if os.path.exists(default_path):
                    path = default_path
                    print(f"Using default dataset: {default_path}")
                else:
                    print("No path provided and default dataset not found.")
                    continue
            df = load_dataset(path)
            if df is not None:
                df = clean_data(df)
        elif choice == "2":
            show_summary_statistics(df)
        elif choice == "3":
            top_n = input("Enter Top-N (or press Enter for all): ").strip()
            show_sales_by_category(df, top_n if top_n else None)
        elif choice == "4":
            top_n = input("Enter Top-N (or press Enter for all): ").strip()
            show_sales_by_month(df, top_n if top_n else None)
        elif choice == "5":
            top_n = input("Enter Top-N (or press Enter for all): ").strip()
            show_ascii_histograms(df, top_n if top_n else None)
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()

