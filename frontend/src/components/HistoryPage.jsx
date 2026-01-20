export default function HistoryPage({ history }) {
    if (history.length === 0) {
        return (
            <div className="card">
                <div className="card-header">
                    <h3 className="card-title">Processing History</h3>
                </div>
                <div className="card-body">
                    <div className="empty-state">
                        <div className="empty-state-icon">📋</div>
                        <p>No images processed yet</p>
                        <p style={{ fontSize: '13px', marginTop: '4px' }}>
                            Upload and enhance images to see them here
                        </p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="card">
            <div className="card-header">
                <h3 className="card-title">Processing History ({history.length} images)</h3>
            </div>
            <div className="card-body">
                <div className="history-grid">
                    {history.map(item => (
                        <div key={item.id} className="history-item">
                            <img src={item.thumbnail} alt={item.filename} className="history-thumb" />
                            <div className="history-info">
                                <h4>{item.filename}</h4>
                                <p>{item.originalSize} → {item.enhancedSize}</p>
                                <p className="history-meta">
                                    {new Date(item.timestamp).toLocaleString()} • {item.processingTime}ms
                                </p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
