import { useState } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import StatsGrid from './components/StatsGrid';
import UploadCard from './components/UploadCard';
import ResultsCard from './components/ResultsCard';
import ActivityPanel from './components/ActivityPanel';
import ModelInfoPanel from './components/ModelInfoPanel';
import HistoryPage from './components/HistoryPage';
import SettingsPage from './components/SettingsPage';
import './styles/Dashboard.css';
import './styles/Upload.css';
import './styles/Panels.css';
import './styles/Pages.css';
import './styles/Analysis.css';
import './styles/Processing.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({
    totalProcessed: 0,
    avgTime: 0,
    totalUpscaled: 0,
    successRate: 100
  });
  const [activities, setActivities] = useState([]);
  const [history, setHistory] = useState([]);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setError(null);
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result);
    reader.readAsDataURL(file);
  };

  const handleEnhance = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_URL}/enhance`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Enhancement failed');
      }

      const data = await response.json();
      setResults(data);

      // Update stats
      setStats(prev => ({
        totalProcessed: prev.totalProcessed + 1,
        avgTime: Math.round((prev.avgTime * prev.totalProcessed + data.processing_time_ms) / (prev.totalProcessed + 1)),
        totalUpscaled: prev.totalUpscaled + 1,
        successRate: 100
      }));

      // Add to history
      const historyItem = {
        id: Date.now(),
        filename: selectedFile.name,
        timestamp: new Date().toISOString(),
        originalSize: `${data.original.width}×${data.original.height}`,
        enhancedSize: `${data.enhanced.width}×${data.enhanced.height}`,
        processingTime: data.processing_time_ms,
        thumbnail: `data:image/png;base64,${data.enhanced.image}`
      };
      setHistory(prev => [historyItem, ...prev]);

      // Add activity
      setActivities(prev => [{
        id: Date.now(),
        title: selectedFile.name,
        status: 'success',
        time: new Date().toLocaleTimeString(),
        duration: `${data.processing_time_ms}ms`
      }, ...prev.slice(0, 4)]);

    } catch (err) {
      setError(err.message);
      setActivities(prev => [{
        id: Date.now(),
        title: selectedFile.name,
        status: 'error',
        time: new Date().toLocaleTimeString(),
        duration: '-'
      }, ...prev.slice(0, 4)]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResults(null);
    setError(null);
  };

  const handleDownload = () => {
    if (!results) return;
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${results.enhanced.image}`;
    link.download = 'enhanced_sar.png';
    link.click();
  };

  const renderPage = () => {
    switch (currentPage) {
      case 'upload':
      case 'dashboard':
        return (
          <>
            <StatsGrid stats={stats} />
            <div className="main-grid">
              <div className="primary-content">
                {!results ? (
                  <UploadCard
                    selectedFile={selectedFile}
                    preview={preview}
                    isProcessing={isProcessing}
                    error={error}
                    onFileSelect={handleFileSelect}
                    onEnhance={handleEnhance}
                    onReset={handleReset}
                  />
                ) : (
                  <ResultsCard
                    results={results}
                    onReset={handleReset}
                    onDownload={handleDownload}
                  />
                )}
              </div>
              <div className="secondary-content">
                <ActivityPanel activities={activities} />
                <ModelInfoPanel />
              </div>
            </div>
          </>
        );
      case 'history':
        return <HistoryPage history={history} />;
      case 'settings':
        return <SettingsPage />;
      default:
        return null;
    }
  };

  return (
    <div className="dashboard">
      <Sidebar currentPage={currentPage} onNavigate={setCurrentPage} />
      <main className="main-content">
        <Header currentPage={currentPage} />
        {renderPage()}
      </main>
    </div>
  );
}

export default App;
