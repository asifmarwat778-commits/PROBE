import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# Publication-ready GBR GUI
# ============================================================

class GBR_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GBR Predictive Model — Ct/C0 Removal Efficiency")
        self.root.geometry("1850x1050")
        self.root.configure(bg='white')

        # Data containers
        self.df = None
        self.model = None
        self.feature_names = None
        self.original_X_cols = None
        self.cat_cols = []
        self.num_cols = []
        self.results = {}
        self.predict_entries = {}

        self._setup_styles()
        self._build_ui()

    # ------------------------------------------------------------------
    # Styling for paper-quality appearance
    # ------------------------------------------------------------------
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Times New Roman', 18, 'bold'), background='white', foreground='#1a1a1a')
        style.configure('Header.TLabelframe.Label', font=('Times New Roman', 12, 'bold'), foreground='#2c3e50')
        style.configure('TButton', font=('Times New Roman', 11), padding=5)
        style.configure('TLabel', font=('Times New Roman', 11), background='white')
        style.configure('TEntry', font=('Times New Roman', 11))
        style.configure('TCombobox', font=('Times New Roman', 11))
        style.configure("Treeview.Heading", font=('Times New Roman', 11, 'bold'))
        style.configure("Treeview", font=('Times New Roman', 10), rowheight=22)
        style.configure("Horizontal.TProgressbar", thickness=20, background='#27ae60')

    # ------------------------------------------------------------------
    # UI Layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        # ===== TOP BANNER =====
        banner = tk.Frame(self.root, bg='#2c3e50', height=60)
        banner.pack(fill=tk.X, side=tk.TOP)
        banner.pack_propagate(False)
        tk.Label(banner, text="Gradient Boosting Regressor (GBR) for Ct/C0 Prediction",
                 font=('Times New Roman', 20, 'bold'), bg='#2c3e50', fg='white').pack(expand=True)

        # ===== MAIN PANED WINDOW =====
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg='white')
        main_pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ===== LEFT CONTROL PANEL =====
        left_frame = tk.Frame(main_pane, bg='white', width=520)
        main_pane.add(left_frame, minsize=500)

        # --- Section 1: Data Input ---
        sec1 = ttk.LabelFrame(left_frame, text=" 1. Dataset Input ", padding=10)
        sec1.pack(fill=tk.X, pady=4, padx=4)

        f1 = tk.Frame(sec1, bg='white')
        f1.pack(fill=tk.X)
        self.path_var = tk.StringVar()
        ttk.Entry(f1, textvariable=self.path_var, width=38).pack(side=tk.LEFT, padx=2)
        ttk.Button(f1, text="Browse CSV", command=self._browse).pack(side=tk.LEFT, padx=2)

        f2 = tk.Frame(sec1, bg='white')
        f2.pack(fill=tk.X, pady=6)
        ttk.Label(f2, text="Target (y):").pack(side=tk.LEFT)
        self.tgt_var = tk.StringVar()
        self.tgt_combo = ttk.Combobox(f2, textvariable=self.tgt_var, width=28, state='readonly')
        self.tgt_combo.pack(side=tk.LEFT, padx=5)

        self.info_var = tk.StringVar(value="No data loaded")
        ttk.Label(sec1, textvariable=self.info_var, foreground='#7f8c8d', font=('Times New Roman', 10, 'italic')).pack(
            anchor=tk.W)

        # --- Section 2: Hyperparameters ---
        sec2 = ttk.LabelFrame(left_frame, text=" 2. Model Hyperparameters ", padding=10)
        sec2.pack(fill=tk.X, pady=4, padx=4)

        params = [
            ('n_estimators', '200', '# trees'),
            ('learning_rate', '0.1', 'shrinkage'),
            ('max_depth', '5', 'tree depth'),
            ('test_size', '0.2', 'fraction'),
            ('random_state', '42', 'seed')
        ]
        self.par_entries = {}
        for i, (lbl, val, hint) in enumerate(params):
            ttk.Label(sec2, text=f"{lbl}:").grid(row=i, column=0, sticky=tk.W, pady=3, padx=2)
            ent = ttk.Entry(sec2, width=12)
            ent.insert(0, val)
            ent.grid(row=i, column=1, sticky=tk.W, pady=3, padx=2)
            self.par_entries[lbl] = ent
            ttk.Label(sec2, text=hint, foreground='#7f8c8d', font=('Times New Roman', 9, 'italic')).grid(row=i,
                                                                                                         column=2,
                                                                                                         sticky=tk.W,
                                                                                                         padx=5)

        # --- Section 3: Training Control ---
        sec3 = ttk.LabelFrame(left_frame, text=" 3. Model Training ", padding=10)
        sec3.pack(fill=tk.X, pady=4, padx=4)

        btn_row = tk.Frame(sec3, bg='white')
        btn_row.pack(fill=tk.X)
        self.train_btn = ttk.Button(btn_row, text="▶  Train GBR Model", command=self._start_training)
        self.train_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.save_btn = ttk.Button(btn_row, text="💾 Save Plots", command=self._save_plots, state='disabled')
        self.save_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.prog = ttk.Progressbar(sec3, mode='indeterminate', length=200)
        self.prog.pack(fill=tk.X, pady=6)

        self.status_var = tk.StringVar(value="Status: Ready")
        ttk.Label(sec3, textvariable=self.status_var, font=('Times New Roman', 10, 'bold'), foreground='#27ae60').pack(
            anchor=tk.W)

        # --- Section 4: Metrics Table ---
        sec4 = ttk.LabelFrame(left_frame, text=" 4. Performance Metrics ", padding=5)
        sec4.pack(fill=tk.X, pady=4, padx=4)

        cols = ('Metric', 'Train', 'Test', 'Combined')
        self.tree = ttk.Treeview(sec4, columns=cols, show='headings', height=7)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=95, anchor='center')
        self.tree.pack(fill=tk.X)

        # --- Section 5: Prediction Interface ---
        sec5 = ttk.LabelFrame(left_frame, text=" 5. Ct/C0 Prediction Interface ", padding=10)
        sec5.pack(fill=tk.BOTH, expand=True, pady=4, padx=4)

        # Scrollable canvas for feature inputs
        canvas_frame = tk.Frame(sec5, bg='white')
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.pred_canvas = tk.Canvas(canvas_frame, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.pred_canvas.yview)
        self.pred_inner = tk.Frame(self.pred_canvas, bg='white')

        self.pred_inner.bind("<Configure>",
                             lambda e: self.pred_canvas.configure(scrollregion=self.pred_canvas.bbox("all")))
        self.pred_canvas.create_window((0, 0), window=self.pred_inner, anchor="nw", width=460)
        self.pred_canvas.configure(yscrollcommand=scrollbar.set)

        self.pred_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Predict button & result (at bottom of sec5, outside scroll)
        ttk.Button(sec5, text="🔮 Predict Ct/C0", command=self._predict_single).pack(fill=tk.X, pady=5)
        self.result_var = tk.StringVar(value="Predicted Ct/C0: —")
        ttk.Label(sec5, textvariable=self.result_var, font=('Times New Roman', 14, 'bold'), foreground='#c0392b').pack(
            anchor=tk.CENTER)

        # ===== RIGHT VISUALIZATION PANEL =====
        right_frame = tk.Frame(main_pane, bg='white')
        main_pane.add(right_frame, minsize=1000)

        self.nb = ttk.Notebook(right_frame)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.tabs = {}
        self.figs = {}
        self.axes = {}
        self.canvases = {}

        for name, title in [('train', 'Training Set'),
                            ('test', 'Test Set'),
                            ('combined', 'Combined Dataset'),
                            ('importance', 'Feature Importance')]:
            tab = tk.Frame(self.nb, bg='white')
            self.nb.add(tab, text=f'  {title}  ')
            self.tabs[name] = tab

            fig = Figure(figsize=(10, 7.5), dpi=110, facecolor='white')
            ax = fig.add_subplot(111)
            self.figs[name] = fig
            self.axes[name] = ax

            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.canvases[name] = canvas

            toolbar = NavigationToolbar2Tk(canvas, tab)
            toolbar.update()

    # ------------------------------------------------------------------
    # File Handling
    # ------------------------------------------------------------------
    def _browse(self):
        p = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not p:
            return
        self.path_var.set(p)
        try:
            self.df = pd.read_csv(p)
            cols = list(self.df.columns)
            self.tgt_combo['values'] = cols
            if 'Ct/C0 (output)' in cols:
                self.tgt_var.set('Ct/C0 (output)')
            else:
                self.tgt_var.set(cols[-1])
            self.info_var.set(f"Loaded: {len(self.df)} rows × {len(cols)} columns  |  File: {p.split('/')[-1]}")
            self.status_var.set("Status: Data loaded. Ready to train.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ------------------------------------------------------------------
    # Training Thread
    # ------------------------------------------------------------------
    def _start_training(self):
        if self.df is None:
            messagebox.showwarning("Missing Data", "Please load a CSV file first.")
            return
        self.train_btn.config(state='disabled')
        self.save_btn.config(state='disabled')
        self.prog.start()
        self.status_var.set("Status: Training in progress...")
        threading.Thread(target=self._train, daemon=True).start()

    def _train(self):
        try:
            target = self.tgt_var.get()
            n_est = int(self.par_entries['n_estimators'].get())
            lr = float(self.par_entries['learning_rate'].get())
            depth = int(self.par_entries['max_depth'].get())
            tsize = float(self.par_entries['test_size'].get())
            rstate = int(self.par_entries['random_state'].get())

            y = self.df[target].values
            X = self.df.drop(columns=[target])

            # Store original columns for prediction interface
            self.original_X_cols = list(X.columns)
            self.cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            self.num_cols = [c for c in X.columns if c not in self.cat_cols]

            # One-hot encoding
            if self.cat_cols:
                X = pd.get_dummies(X, columns=self.cat_cols, drop_first=False)

            self.feature_names = list(X.columns)
            X = X.values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=tsize, random_state=rstate
            )

            gbr = GradientBoostingRegressor(
                n_estimators=n_est, learning_rate=lr, max_depth=depth, random_state=rstate
            )
            gbr.fit(X_train, y_train)
            self.model = gbr

            p_train = gbr.predict(X_train)
            p_test = gbr.predict(X_test)
            p_all = gbr.predict(X)

            self.results = {
                'train': (y_train, p_train, self._metrics(y_train, p_train)),
                'test': (y_test, p_test, self._metrics(y_test, p_test)),
                'combined': (y, p_all, self._metrics(y, p_all)),
                'importance': gbr.feature_importances_
            }

            self.root.after(0, self._render_all)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Training Failed", str(e)))
            self.root.after(0, self._reset_ui)

    # ------------------------------------------------------------------
    # Metrics Computation
    # ------------------------------------------------------------------
    def _metrics(self, obs, pred):
        obs, pred = np.asarray(obs), np.asarray(pred)
        mo = np.mean(obs)

        # Willmott's d
        num = np.sum((obs - pred) ** 2)
        den = np.sum((np.abs(pred - mo) + np.abs(obs - mo)) ** 2)
        d = 1.0 if den == 0 else 1 - num / den

        # KGE
        r = np.corrcoef(obs, pred)[0, 1]
        alpha = np.std(pred) / np.std(obs) if np.std(obs) != 0 else 1
        beta = np.mean(pred) / np.mean(obs) if np.mean(obs) != 0 else 1
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

        # E1
        num1 = np.sum(np.abs(obs - pred))
        den1 = np.sum(np.abs(obs - mo))
        e1 = 1.0 if den1 == 0 else 1 - num1 / den1

        return {
            'R2': r2_score(obs, pred),
            'd': d,
            'KGE': kge,
            'E1': e1,
            'MAE': mean_absolute_error(obs, pred),
            'RMSE': np.sqrt(mean_squared_error(obs, pred)),
            'MBE': np.mean(pred - obs)
        }

    # ------------------------------------------------------------------
    # Render Results
    # ------------------------------------------------------------------
    def _render_all(self):
        self.prog.stop()
        self.train_btn.config(state='normal')
        self.save_btn.config(state='normal')
        self.status_var.set("Status: Training complete. Model ready.")

        # Metrics table
        for item in self.tree.get_children():
            self.tree.delete(item)
        names = ['R²', "Willmott's d", 'KGE', 'E₁', 'MAE', 'RMSE', 'MBE']
        keys = ['R2', 'd', 'KGE', 'E1', 'MAE', 'RMSE', 'MBE']
        for n, k in zip(names, keys):
            self.tree.insert('', tk.END, values=(
                n,
                f"{self.results['train'][2][k]:.5f}",
                f"{self.results['test'][2][k]:.5f}",
                f"{self.results['combined'][2][k]:.5f}"
            ))

        # Plots
        self._plot_scatter('train', 'GBR: Predicted vs Actual (Training Set)')
        self._plot_scatter('test', 'GBR: Predicted vs Actual (Test Set)')
        self._plot_scatter('combined', 'GBR: Predicted vs Actual (Combined Dataset)')
        self._plot_importance()

        # Build prediction interface
        self._build_predictor()

    def _plot_scatter(self, key, title):
        fig, ax = self.figs[key], self.axes[key]
        ax.clear()
        obs, pred, met = self.results[key]

        ax.scatter(obs, pred, c='#2980b9', edgecolors='k', alpha=0.75, s=90, zorder=3, label='Data points')
        lo, hi = min(obs.min(), pred.min()), max(obs.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=2.2, label='1:1 line', zorder=2)

        ax.set_xlabel('Actual Ct/C$_0$', fontsize=13, fontname='Times New Roman')
        ax.set_ylabel('Predicted Ct/C$_0$', fontsize=13, fontname='Times New Roman')
        ax.set_title(title, fontsize=15, fontname='Times New Roman', pad=12)

        txt = (f"R$^2$ = {met['R2']:.4f}\n"
               f"$d$ = {met['d']:.4f}\n"
               f"KGE = {met['KGE']:.4f}\n"
               f"E$_1$ = {met['E1']:.4f}\n"
               f"MAE = {met['MAE']:.4f}\n"
               f"RMSE = {met['RMSE']:.4f}\n"
               f"MBE = {met['MBE']:.4f}")
        ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontname='Times New Roman',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#ecf0f1', edgecolor='#bdc3c7', alpha=0.95))

        ax.legend(loc='lower right', fontsize=11, prop={'family': 'Times New Roman'})
        ax.grid(True, linestyle='--', alpha=0.4, zorder=1)
        ax.set_axisbelow(True)
        fig.tight_layout()
        self.canvases[key].draw()

    def _plot_importance(self):
        fig, ax = self.figs['importance'], self.axes['importance']
        ax.clear()
        imp = self.results['importance']
        idx = np.argsort(imp)[::-1][:min(15, len(imp))]

        colors = plt.cm.RdYlBu(np.linspace(0.2, 0.8, len(idx)))[::-1]
        ax.barh(range(len(idx)), imp[idx], color=colors, edgecolor='k', height=0.7)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([self.feature_names[i] for i in idx], fontsize=10, fontname='Times New Roman')
        ax.invert_yaxis()
        ax.set_xlabel('Relative Importance', fontsize=13, fontname='Times New Roman')
        ax.set_title('Feature Importance Ranking (GBR)', fontsize=15, fontname='Times New Roman', pad=12)
        ax.grid(True, axis='x', linestyle='--', alpha=0.4)
        fig.tight_layout()
        self.canvases['importance'].draw()

    # ------------------------------------------------------------------
    # Prediction Interface Builder
    # ------------------------------------------------------------------
    def _build_predictor(self):
        # Clear old
        for widget in self.pred_inner.winfo_children():
            widget.destroy()
        self.predict_entries.clear()

        ttk.Label(self.pred_inner, text="Enter input values for prediction:",
                  font=('Times New Roman', 11, 'bold')).pack(anchor=tk.W, pady=(0, 6))

        # Numeric inputs
        for col in self.num_cols:
            row = tk.Frame(self.pred_inner, bg='white')
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{col}:", width=22, anchor='e').pack(side=tk.LEFT)
            ent = ttk.Entry(row, width=18)
            ent.insert(0, "0.0")
            ent.pack(side=tk.LEFT, padx=4)
            self.predict_entries[col] = ent

        # Categorical inputs
        for col in self.cat_cols:
            row = tk.Frame(self.pred_inner, bg='white')
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{col}:", width=22, anchor='e').pack(side=tk.LEFT)
            unique_vals = sorted(self.df[col].dropna().unique().tolist())
            combo = ttk.Combobox(row, values=unique_vals, width=16, state='readonly')
            if unique_vals:
                combo.set(unique_vals[0])
            combo.pack(side=tk.LEFT, padx=4)
            self.predict_entries[col] = combo

        self.pred_canvas.update_idletasks()
        self.pred_canvas.configure(scrollregion=self.pred_canvas.bbox("all"))
        self.result_var.set("Predicted Ct/C0: — (enter values above)")

    def _predict_single(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Train the model first.")
            return
        try:
            # Build 1-row dataframe from entries
            input_dict = {}
            for col, widget in self.predict_entries.items():
                if isinstance(widget, ttk.Combobox):
                    input_dict[col] = widget.get()
                else:
                    input_dict[col] = float(widget.get())

            input_df = pd.DataFrame([input_dict])

            # One-hot encode identically to training
            if self.cat_cols:
                input_df = pd.get_dummies(input_df, columns=self.cat_cols, drop_first=False)

            # Align columns
            input_df = input_df.reindex(columns=self.feature_names, fill_value=0)
            X_pred = input_df.values

            pred = self.model.predict(X_pred)[0]
            self.result_var.set(f"Predicted Ct/C0: {pred:.5f}")
            self.status_var.set(f"Status: Single prediction complete ({pred:.5f})")

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _save_plots(self):
        if not self.results:
            return
        folder = filedialog.askdirectory(title="Select folder to save figures")
        if not folder:
            return
        for k in ['train', 'test', 'combined', 'importance']:
            self.figs[k].savefig(f"{folder}/GBR_{k.capitalize()}.png", dpi=300, bbox_inches='tight')
        messagebox.showinfo("Saved", f"Figures exported to:\n{folder}")

    def _reset_ui(self):
        self.prog.stop()
        self.train_btn.config(state='normal')


# ====================================================================
if __name__ == '__main__':
    root = tk.Tk()
    app = GBR_GUI(root)
    root.mainloop()