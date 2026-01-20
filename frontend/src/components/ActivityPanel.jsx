export default function ActivityPanel({ activities }) {
    return (
        <div className="card activity-panel">
            <div className="card-header">
                <h3 className="card-title">Recent Activity</h3>
            </div>
            <div className="card-body">
                {activities.length === 0 ? (
                    <div className="empty-state">
                        <div className="empty-state-icon">📋</div>
                        <p>No activity yet</p>
                    </div>
                ) : (
                    <div className="activity-list">
                        {activities.map((activity) => (
                            <div key={activity.id} className="activity-item">
                                <div className={`activity-icon ${activity.status}`}>
                                    {activity.status === 'success' ? '✓' : '✕'}
                                </div>
                                <div className="activity-content">
                                    <div className="activity-title">{activity.title}</div>
                                    <div className="activity-meta">
                                        {activity.time} • {activity.duration}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
