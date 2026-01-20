# RadarVision AI - SAR Image Super-Resolution

A web-based application for **resolution enhancement of automotive SAR (Synthetic Aperture Radar) images** using deep learning super-resolution.

![Dashboard Preview](https://via.placeholder.com/800x450/0a0a0f/00f5d4?text=RadarVision+AI+Dashboard)

## 🚀 Features

- **4x Super-Resolution** - Quadruple your SAR image resolution using ESPCN neural network
- **Real-Time Processing** - Optimized CNN architecture for fast inference
- **Interactive Comparison** - Side-by-side slider to compare original vs enhanced
- **Professional Dashboard** - Modern, dark-themed UI with live statistics
- **Drag & Drop Upload** - Easy file upload supporting PNG, JPEG, TIFF formats
- **REST API** - FastAPI backend with Swagger documentation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React + Vite)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Dashboard│  │  Upload  │  │ Results  │  │  Stats   │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP/REST
┌─────────────────────────▼───────────────────────────────────┐
│                   Backend (FastAPI + PyTorch)                │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┐  │
│  │   API    │  │  ESPCN   │  │  Image Pre/Post Process  │  │
│  │ Endpoints│  │  Model   │  │                          │  │
│  └──────────┘  └──────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 🧠 Model: ESPCN

**Efficient Sub-Pixel Convolutional Neural Network**

- Lightweight CNN architecture optimized for real-time processing
- Sub-pixel convolution for efficient upscaling
- Trained for grayscale SAR/radar imagery
- 4x resolution enhancement

```
Input (LR) → Conv(64) → Conv(32) → PixelShuffle(4x) → Output (HR)
```

## 📦 Project Structure

```
Radar/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── requirements.txt     # Python dependencies
│   └── models/
│       ├── __init__.py
│       └── sr_model.py      # ESPCN model implementation
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React component
│   │   ├── main.jsx         # Entry point
│   │   ├── index.css        # Global styles
│   │   ├── styles/          # Component styles
│   │   └── components/      # React components
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn

### 1. Start the Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- API info: `http://localhost:8000/model/info`

### 2. Start the Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The dashboard will be available at `http://localhost:5173`

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API health check and info |
| GET | `/health` | Health check |
| GET | `/model/info` | Model architecture details |
| POST | `/enhance` | Enhance image (returns base64) |
| POST | `/enhance/download` | Enhance and download image |

### Example API Call

```bash
curl -X POST "http://localhost:8000/enhance" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_sar_image.png"
```

## 🎯 Use Cases

### Autonomous Vehicles
- Enhance low-resolution SAR captures for better object detection
- Improve radar-based perception systems
- Real-time processing for navigation systems

### Research & Development
- Experiment with super-resolution techniques
- Compare bicubic vs AI upscaling
- Dataset preparation and augmentation

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | React 18, Vite |
| Styling | Vanilla CSS with CSS Variables |
| Backend | FastAPI, Uvicorn |
| ML Framework | PyTorch |
| Model | ESPCN (CNN-based) |
| Image Processing | Pillow, OpenCV, NumPy |

## 🔧 Configuration

### Backend Environment Variables

```bash
# Optional: Set host and port
HOST=0.0.0.0
PORT=8000
```

### Frontend API URL

Edit `src/App.jsx`:
```javascript
const API_URL = 'http://localhost:8000';  // Change for production
```

## 📈 Future Enhancements

- [ ] Add GAN-based super-resolution (SRGAN, ESRGAN)
- [ ] Batch processing support
- [ ] GPU acceleration (CUDA)
- [ ] Pre-trained weights for SAR imagery
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP)

## 📄 License

MIT License - feel free to use for your autonomous vehicle projects!

---

Built with ❤️ for the automotive AI community
# radar_pbl
