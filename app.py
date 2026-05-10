import sys
import pandas as pd
import joblib
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QSlider, QCheckBox,
                             QComboBox, QPushButton, QTextEdit, QFrame)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class QyzylordaAIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qyzylorda House Price Prediction Model")
        self.setMinimumSize(1000, 750)

        # Модельді жүктеу
        try:
            self.model = joblib.load('models/house_price_model.pkl')
        except:
            print("Модель файлы табылмады!")

        # СТИЛЬ (Modern Dark UI)
        self.setStyleSheet("""
            QMainWindow { background-color: #0e1117; }
            QFrame#Block { 
                background-color: #161b22; 
                border-radius: 20px; 
                border: 1px solid #30363d; 
            }
            QLabel { color: #e6edf3; font-family: 'Segoe UI'; font-size: 14px; }
            QLabel#Header { color: #00d4ff; font-size: 26px; font-weight: bold; }
            QLabel#Price { color: #00d4ff; font-size: 50px; font-weight: bold; }
            QPushButton { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00d4ff, stop:1 #0055ff);
                color: white; border-radius: 12px; font-weight: bold; height: 50px; font-size: 16px;
            }
            QPushButton:hover { background: #00b4d8; }
            QComboBox, QLineEdit { 
                background-color: #0d1117; color: white; border: 1px solid #30363d; 
                border-radius: 8px; padding: 8px; font-size: 14px;
            }
            QCheckBox { color: white; font-weight: bold; }
            QSlider::handle:horizontal { background: #00d4ff; width: 18px; border-radius: 9px; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # HEADER
        header_lbl = QLabel("🏙️ Qyzylorda House Price Prediction Model")
        header_lbl.setObjectName("Header")
        header_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header_lbl)

        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        # --- BLOCK 1: ПАРАМЕТРЛЕР ---
        self.block1 = QFrame();
        self.block1.setObjectName("Block")
        b1_layout = QVBoxLayout(self.block1)
        b1_layout.addWidget(QLabel("📋 НЫСАН СИПАТТАМАСЫ", font=QFont("Arial", 12, QFont.Weight.Bold)))

        self.is_house = QCheckBox("🏡 Бұл жеке жер үй")
        self.is_house.stateChanged.connect(self.toggle_ui)
        b1_layout.addWidget(self.is_house)

        self.area_input = QLineEdit("85")
        b1_layout.addWidget(QLabel("Ауданы (м²):"));
        b1_layout.addWidget(self.area_input)

        self.rooms_slider = QSlider(Qt.Orientation.Horizontal)
        self.rooms_slider.setRange(1, 10);
        self.rooms_slider.setValue(3)
        self.rooms_lbl = QLabel("Бөлме саны: 3")
        self.rooms_slider.valueChanged.connect(lambda v: self.rooms_lbl.setText(f"Бөлме саны: {v}"))
        b1_layout.addWidget(self.rooms_lbl);
        b1_layout.addWidget(self.rooms_slider)

        self.mat_menu = QComboBox()
        self.mat_menu.addItems(["Кирпич", "Панель", "Бетон"])
        b1_layout.addWidget(QLabel("Үй материалы:"));
        b1_layout.addWidget(self.mat_menu)

        self.rep_menu = QComboBox()
        self.rep_menu.addItems(["Черновой", "Орташа", "Еуро"])
        self.rep_menu.setCurrentText("Орташа")
        b1_layout.addWidget(QLabel("Жөндеу деңгейі:"));
        b1_layout.addWidget(self.rep_menu)

        content_layout.addWidget(self.block1)

        # --- BLOCK 2: ЛОКАЦИЯ ЖӘНЕ ЭКОНОМИКА ---
        self.block2 = QFrame();
        self.block2.setObjectName("Block")
        b2_layout = QVBoxLayout(self.block2)
        b2_layout.addWidget(QLabel("📍 ЛОКАЦИЯ ЖӘНЕ ВАЛЮТА", font=QFont("Arial", 12, QFont.Weight.Bold)))

        self.districts_db = {
            "Орталық": {"m": 1.35, "h": False, "a": True}, "Сырдария": {"m": 1.28, "h": False, "a": True},
            "ЖК Мерей": {"m": 1.30, "h": False, "a": True}, "Сол Жағалау": {"m": 1.32, "h": False, "a": True},
            "Шұғыла": {"m": 1.18, "h": True, "a": True}, "Микр. Байтерек": {"m": 1.10, "h": False, "a": True},
            "Универсам": {"m": 1.12, "h": False, "a": True}, "Арай": {"m": 1.15, "h": True, "a": False},
            "Ақмаржан": {"m": 1.08, "h": False, "a": True}, "Сәулет": {"m": 0.98, "h": False, "a": True},
            "Микр. Мерей": {"m": 1.05, "h": False, "a": True}, "Титов": {"m": 0.85, "h": True, "a": True}
        }
        self.dist_menu = QComboBox()
        self.dist_menu.addItems(self.districts_db.keys())
        b2_layout.addWidget(QLabel("Ауданды таңдаңыз:"));
        b2_layout.addWidget(self.dist_menu)

        # Динамикалық бөлім
        self.dyn_stack = QFrame()
        dyn_lay = QVBoxLayout(self.dyn_stack)
        self.f_slider = QSlider(Qt.Orientation.Horizontal);
        self.f_slider.setRange(1, 5);
        self.f_slider.setValue(3)
        self.f_lbl = QLabel("Пәтер қабаты: 3")
        self.f_slider.valueChanged.connect(lambda v: self.f_lbl.setText(f"Пәтер қабаты: {v}"))
        self.l_slider = QSlider(Qt.Orientation.Horizontal);
        self.l_slider.setRange(1, 20);
        self.l_slider.setValue(6)
        self.l_lbl = QLabel("Жер көлемі: 6 сотка")
        self.l_slider.valueChanged.connect(lambda v: self.l_lbl.setText(f"Жер көлемі: {v} сотка"))
        dyn_lay.addWidget(self.f_lbl);
        dyn_lay.addWidget(self.f_slider)
        dyn_lay.addWidget(self.l_lbl);
        dyn_lay.addWidget(self.l_slider)
        self.l_lbl.hide();
        self.l_slider.hide();
        b2_layout.addWidget(self.dyn_stack)

        self.shock_slider = QSlider(Qt.Orientation.Horizontal)
        self.shock_slider.setRange(400, 850);
        self.shock_slider.setValue(450)
        self.shock_lbl = QLabel("Болжамды курс: 450 ₸")
        self.shock_slider.valueChanged.connect(lambda v: self.shock_lbl.setText(f"Болжамды курс: {v} ₸"))
        b2_layout.addWidget(self.shock_lbl);
        b2_layout.addWidget(self.shock_slider)

        self.income_input = QLineEdit("650000")
        b2_layout.addWidget(QLabel("Айлық табыс (₸):"));
        b2_layout.addWidget(self.income_input)

        content_layout.addWidget(self.block2)

        # --- BLOCK 3: НӘТИЖЕ ---
        self.block3 = QFrame();
        self.block3.setObjectName("Block")
        self.block3.setStyleSheet("border: 2px solid #00d4ff; background-color: #1a1c23;")
        b3_layout = QVBoxLayout(self.block3)
        b3_layout.addWidget(QLabel("🎯 ЕСЕПТЕУ НӘТИЖЕСІ", alignment=Qt.AlignmentFlag.AlignCenter))
        self.price_val = QLabel("0 ₸");
        self.price_val.setObjectName("Price")
        self.price_val.setAlignment(Qt.AlignmentFlag.AlignCenter);
        b3_layout.addWidget(self.price_val)
        self.inf_lbl = QLabel("+0% инфляция", alignment=Qt.AlignmentFlag.AlignCenter)
        self.inf_lbl.setStyleSheet("color: #ff4b4b; font-weight: bold; font-size: 16px;");
        b3_layout.addWidget(self.inf_lbl)
        self.calc_btn = QPushButton("АНАЛИЗ ЖАСАУ");
        self.calc_btn.clicked.connect(self.calculate);
        b3_layout.addWidget(self.calc_btn)
        self.report = QTextEdit();
        self.report.setReadOnly(True)
        self.report.setStyleSheet("background-color: #0d1117; border: none; font-family: 'Consolas';");
        b3_layout.addWidget(self.report)
        content_layout.addWidget(self.block3)

    def toggle_ui(self):
        is_h = self.is_house.isChecked()
        self.f_lbl.setVisible(not is_h);
        self.f_slider.setVisible(not is_h)
        self.l_lbl.setVisible(is_h);
        self.l_slider.setVisible(is_h)

    def calculate(self):
        try:
            sel_d = self.dist_menu.currentText();
            info = self.districts_db[sel_d];
            is_h = self.is_house.isChecked()
            if is_h and not info["h"]: self.report.setText(f"❌ {sel_d} ауданында ЖЕР ҮЙ жоқ!"); return
            if not is_h and not info["a"]: self.report.setText(f"❌ {sel_d} ауданында ПӘТЕР жоқ!"); return
            area = float(self.area_input.text());
            USD = 450;
            MULT = 0.6;
            QYZ = 0.4
            future_usd = self.shock_slider.value()
            mat_m = {"Кирпич": 1.15, "Панель": 0.95, "Бетон": 1.10}[self.mat_menu.currentText()]
            rep_m = {"Черновой": 0.8, "Орташа": 1.0, "Еуро": 1.3}[self.rep_menu.currentText()]
            f_imp = 1.15 if (not is_h and self.f_slider.value() in [2, 3, 4]) else 0.9
            med_inc = (float(self.income_input.text()) * 12) / USD / 10000
            inp = pd.DataFrame({'MedInc': [med_inc], 'HouseAge': [20], 'AveRooms': [area / 25], 'AveBedrms': [1.2],
                                'Population': [1500], 'AveOccup': [3.5], 'Latitude': [34.0], 'Longitude': [-118.0]})
            raw_p = self.model.predict(inp)[0]
            p_base = raw_p * 100000 * USD * MULT * info["m"] * QYZ * f_imp * mat_m * rep_m
            p_shock = raw_p * 100000 * future_usd * MULT * info["m"] * QYZ * f_imp * mat_m * rep_m
            if is_h: p_base += (self.l_slider.value() * 1500000); p_shock += (self.l_slider.value() * 1500000)
            inf_perc = ((p_shock - p_base) / p_base) * 100
            self.price_val.setText(f"{int(p_shock):,} ₸")
            self.inf_lbl.setText(f"+{int(inf_perc)}% инфляция")
            self.report.setText(
                f"AI САРАПТАМА:\nАудан: {sel_d}\nТүрі: {'Жер үй' if is_h else 'Пәтер'}\n1 м2 құны: {int(p_shock / area):,} ₸")
        except Exception as e:
            self.report.setText(f"Қате: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv);
    window = QyzylordaAIApp();
    window.show();
    sys.exit(app.exec())