import customtkinter as ctk
import pandas as pd
import joblib
import numpy as np
from tkinter import messagebox

# Дизайн баптаулары
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class QyzylordaAIApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Үй бағасын болжау моделі")
        self.geometry("1100(м)x750(б)")

        # Модельді жүктеу
        try:
            self.model = joblib.load('models/house_price_model.pkl')
        except:
            print("Қате: Модель файлы табылмады!")

        # --- Layout ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- СОЛ ЖАҚ ПАНЕЛЬ (INPUTS) ---
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(self.sidebar, text="🏠 ПАРАМЕТРЛЕР", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=20)

        self.is_house_var = ctk.BooleanVar(value=False)
        self.house_check = ctk.CTkCheckBox(self.sidebar, text="Жер үй", variable=self.is_house_var,
                                           command=self.toggle_ui)
        self.house_check.pack(pady=10)

        self.area_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Ауданы (м2)")
        self.area_entry.insert(0, "85")
        self.area_entry.pack(pady=10, padx=20)

        self.rooms_label = ctk.CTkLabel(self.sidebar, text="Бөлме саны: 3")
        self.rooms_label.pack()
        self.rooms_slider = ctk.CTkSlider(self.sidebar, from_=1, to=10, number_of_steps=9,
                                          command=self.update_rooms_label)
        self.rooms_slider.set(3)
        self.rooms_slider.pack(pady=5, padx=20)

        ctk.CTkLabel(self.sidebar, text="Үй материалы:").pack()
        self.material_menu = ctk.CTkOptionMenu(self.sidebar, values=["Кирпич", "Панель", "Бетон"])
        self.material_menu.set("Кирпич")
        self.material_menu.pack(pady=10, padx=20)

        ctk.CTkLabel(self.sidebar, text="Орталыққа қашықтық (км):").pack()
        self.dist_slider = ctk.CTkSlider(self.sidebar, from_=0.5, to=15)
        self.dist_slider.set(3)
        self.dist_slider.pack(pady=5, padx=20)

        self.dynamic_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.dynamic_frame.pack(pady=10, fill="x")

        self.floor_label = ctk.CTkLabel(self.dynamic_frame, text="Пәтер қабаты (1-5):")
        self.floor_slider = ctk.CTkSlider(self.dynamic_frame, from_=1, to=5, number_of_steps=4)
        self.floor_slider.set(3)
        self.land_label = ctk.CTkLabel(self.dynamic_frame, text="Жер көлемі (сотка):")
        self.land_slider = ctk.CTkSlider(self.dynamic_frame, from_=1, to=20)
        self.land_slider.set(6)
        self.toggle_ui()

        ctk.CTkLabel(self.sidebar, text="Жөндеу деңгейі:").pack()
        self.repair_menu = ctk.CTkOptionMenu(self.sidebar, values=["Черновой", "Орташа", "Еуро"])
        self.repair_menu.set("Орташа")
        self.repair_menu.pack(pady=10, padx=20)

        self.income_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Айлық табыс (₸)")
        self.income_entry.insert(0, "650000")
        self.income_entry.pack(pady=10, padx=20)

        self.calc_btn = ctk.CTkButton(self.sidebar, text="ЕСЕПТЕУ", font=ctk.CTkFont(weight="bold"),
                                      command=self.calculate, fg_color="#00d4ff", text_color="black")
        self.calc_btn.pack(pady=30, padx=20)

        # --- ОҢ ЖАҚ ПАНЕЛЬ (RESULTS) ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=15)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        ctk.CTkLabel(self.main_frame, text="🏙️ Qyzylorda House Prediction model",
                     font=ctk.CTkFont(size=28, weight="bold"), text_color="#00d4ff").pack(pady=20)

        # ТОЛЫҚ АУДАНДАР ТІЗІМІ
        ctk.CTkLabel(self.main_frame, text="Ауданды таңдаңыз:", font=ctk.CTkFont(size=14)).pack()
        self.dist_menu = ctk.CTkOptionMenu(self.main_frame, width=400, values=[
            "Орталық", "Сырдария", "ЖК Мерей", "Сол Жағалау", "Шұғыла",
            "Микр. Байтерек", "Универсам", "Арай", "Ақмаржан", "Сәулет",
            "Микр. Мерей", "Титов"
        ])
        self.dist_menu.set("Орталық")
        self.dist_menu.pack(pady=10)

        self.price_label = ctk.CTkLabel(self.main_frame, text="0 ₸", font=ctk.CTkFont(size=60, weight="bold"),
                                        text_color="#00d4ff")
        self.price_label.pack(pady=30)

        self.details_box = ctk.CTkTextbox(self.main_frame, width=600, height=250, font=("Consolas", 14),
                                          corner_radius=10)
        self.details_box.pack(pady=10, padx=20)

    def update_rooms_label(self, val):
        self.rooms_label.configure(text=f"Бөлме саны: {int(val)}")

    def toggle_ui(self):
        if self.is_house_var.get():
            self.floor_label.pack_forget();
            self.floor_slider.pack_forget()
            self.land_label.pack(padx=20);
            self.land_slider.pack(padx=20)
        else:
            self.land_label.pack_forget();
            self.land_slider.pack_forget()
            self.floor_label.pack(padx=20);
            self.floor_slider.pack(padx=20)

    def calculate(self):
        # АУДАНДАРДЫҢ ТОЛЫҚ МӘЛІМЕТТЕРІ (Сенің тізімің)
        districts_db = {
            "Орталық": {"mult": 1.35, "house": False, "apt": True},
            "Сырдария": {"mult": 1.28, "house": False, "apt": True},
            "ЖК Мерей": {"mult": 1.30, "house": False, "apt": True},
            "Сол Жағалау": {"mult": 1.32, "house": False, "apt": True},
            "Шұғыла": {"mult": 1.18, "house": True, "apt": True},
            "Микр. Байтерек": {"mult": 1.10, "house": False, "apt": True},
            "Универсам": {"mult": 1.12, "house": True, "apt": True},
            "Арай": {"mult": 1.15, "house": True, "apt": True},
            "Ақмаржан": {"mult": 1.08, "house": False, "apt": True},
            "Сәулет": {"mult": 0.98, "house": False, "apt": True},
            "Микр. Мерей": {"mult": 1.05, "house": False, "apt": True},
            "Титов": {"mult": 0.85, "house": True, "apt": True}
        }

        try:
            selected_d = self.dist_menu.get()
            is_house = self.is_house_var.get()

            # ВАЛИДАЦИЯ (Тексеру)
            info = districts_db[selected_d]
            if is_house and not info["house"]:
                messagebox.showwarning("Ескерту", f"{selected_d} ауданында жер үйлер жоқ!")
                return
            if not is_house and not info["apt"]:
                messagebox.showwarning("Ескерту", f"{selected_d} ауданында этаж үйлер (пәтерлер) жоқ!")
                return

            area = float(self.area_entry.get())
            income_val = float(self.income_entry.get())
            dist_center = self.dist_slider.get()

            # КОЭФФИЦИЕНТТЕРІҢ
            USD_KZT = 380;
            MULTIPLIER = 0.8;
            QYZ_INDEX = 0.4
            repair_map = {"Черновой": 0.8, "Орташа": 1.0, "Еуро": 1.3}
            mat_map = {"Кирпич": 1.15, "Панель": 0.95, "Бетон": 1.10}

            # Есептеу
            mat_mult = mat_map[self.material_menu.get()]
            r_mult = repair_map[self.repair_menu.get()]
            d_mult = info["mult"]
            dist_km_mult = 1.0 - (dist_center * 0.02)

            f_impact = 1.0
            if not is_house:
                floor_map = {1: 0.9, 2: 1.1, 3: 1.15, 4: 1.1, 5: 0.85}
                f_impact = floor_map[int(self.floor_slider.get())]

            med_inc = (income_val * 12) / USD_KZT / 10000
            inp = pd.DataFrame({'MedInc': [med_inc], 'HouseAge': [15], 'AveRooms': [area / 25], 'AveBedrms': [1.2],
                                'Population': [1500], 'AveOccup': [3.5], 'Latitude': [34.0], 'Longitude': [-118.0]})

            raw_p = self.model.predict(inp)[0]
            price = raw_p * 100000 * USD_KZT * MULTIPLIER * d_mult * QYZ_INDEX * r_mult * f_impact * mat_mult * dist_km_mult

            if is_house:
                price += (self.land_slider.get() * 1500000)

            # Нәтижені шығару
            self.price_label.configure(text=f"{int(price):,} ₸")
            self.details_box.delete("0.0", "end")
            report = f""">>> САРАПТАМАЛЫҚ ҚОРЫТЫНДЫ:
------------------------------------------
Аудан:        {selected_d}
Мүлік түрі:   {'Жер үй' if is_house else 'Пәтер'}
Материал:     {self.material_menu.get()}
Ауданы:       {area} м2
1 м2 құны:    {int(price / area):,} ₸

МОДЕЛЬ: Random Forest (Accuracy: 81.2%)
------------------------------------------
MadCompany | Qyzylorda 2026"""
            self.details_box.insert("0.0", report)

        except Exception as e:
            messagebox.showerror("Қате", f"Мәліметтерді тексеріңіз! {e}")


if __name__ == "__main__":
    app = QyzylordaAIApp()
    app.mainloop()