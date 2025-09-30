# 🚗 Parking Space Counter

This project is a **real-time parking space counter** built with [YOLO](https://github.com/ultralytics/ultralytics) for object detection and [Streamlit](https://streamlit.io/) for interactive visualization.
It detects **empty** and **occupied** parking spaces from a video feed or camera stream and displays the results with live annotations.

---

## ✨ Features

* Real-time detection of cars using YOLO.
* Counts **Empty** vs **Occupied** spaces.
* Displays annotated frames with results in a Streamlit dashboard.
* Easy to run locally or deploy on the cloud.

---

## 🌐 Live Demo

Try it out here 👉 [Parking Space Counter App](https://parking-space-counter.streamlit.app/)

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/parking-space-counter.git
cd parking-space-counter
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you're running on Streamlit Cloud or Hugging Face Spaces, make sure to include `packages.txt` with:

```
libgl1
libglib2.0-0
```

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Upload a video or connect a webcam feed to see detection in action.

---

## 📸 Example Output

Annotated frame with counters displayed at the top:

```
Empty: 12 | Occupied: 8
```

---

## 📂 Project Structure

```
parking-space-counter/
│── app.py                # Streamlit application
│── requirements.txt       # Python dependencies
│── packages.txt           # System dependencies (for cloud deploys)
│── models/                # YOLO model weights
│── data/                  # Sample videos or images
```

---

## 🔧 Tech Stack

* **Python 3.10+**
* **YOLO (Ultralytics)** for object detection
* **OpenCV** for image processing
* **Streamlit** for UI/dashboard

---

## 🌍 Deployment

* Can be deployed on **Streamlit Cloud** or **Hugging Face Spaces**.
* Make sure both `requirements.txt` and `packages.txt` are present.

---

## 📜 License

This project is licensed under the MIT License.

---

👨‍💻 Developed by **[ENG Muhammad Al-Aasar]**
