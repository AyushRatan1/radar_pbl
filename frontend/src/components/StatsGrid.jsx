export default function StatsGrid({ stats }) {
    return (
        <div className="stats-grid">
            <div className="stat-card">
                <div className="stat-header">
                    <span className="stat-label">Images Processed</span>
                    <div className="stat-icon cyan">📊</div>
                </div>
                <div className="stat-value">{stats.totalProcessed}</div>
                <div className="stat-change">This session</div>
            </div>

            <div className="stat-card">
                <div className="stat-header">
                    <span className="stat-label">Avg. Processing Time</span>
                    <div className="stat-icon purple">⚡</div>
                </div>
                <div className="stat-value">{stats.avgTime}<span style={{ fontSize: '14px', fontWeight: 400 }}>ms</span></div>
                <div className="stat-change">Per image</div>
            </div>

            <div className="stat-card">
                <div className="stat-header">
                    <span className="stat-label">Upscale Factor</span>
                    <div className="stat-icon blue">🔍</div>
                </div>
                <div className="stat-value">4x</div>
                <div className="stat-change">Resolution</div>
            </div>

            <div className="stat-card">
                <div className="stat-header">
                    <span className="stat-label">Model</span>
                    <div className="stat-icon orange">🧠</div>
                </div>
                <div className="stat-value" style={{ fontSize: '20px' }}>ESPCN</div>
                <div className="stat-change">CNN-based</div>
            </div>
        </div>
    );
}
