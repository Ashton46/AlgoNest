import React, { useState, useEffect } from 'react';
import { Activity } from 'lucide-react';

const InteractiveBasketballCourt = ({ prediction, formData }) => {
  const [shotChart, setShotChart] = useState([]);

  useEffect(() => {
    const zones = [
      { name: 'Paint', x: 50, y: 15, probability: null, color: '#ff6262ff' },
      { name: 'Mid-Range Left', x: 30, y: 35, probability: null, color: '#4ecdc4' },
      { name: 'Mid-Range Right', x: 70, y: 35, probability: null, color: '#c14be1ff' },
      { name: '3PT Corner Left', x: 15, y: 85, probability: null, color: '#1cdd83ff' },
      { name: '3PT Corner Right', x: 85, y: 85, probability: null, color: '#e5db1dff' },
      { name: '3PT Top', x: 50, y: 75, probability: null, color: '#ff9ff3' },
    ];
    setShotChart(zones);
  }, []);

  return (
    <div className="basketball-court-container">
      <div className="court-header">
        <Activity size={24} />
        <h3>Shot Probability Chart</h3>
      </div>
      
      <div className="court-svg-container">
        <svg viewBox="0 0 500 470" className="basketball-court">
          {/*Court outline*/}
          <rect x="25" y="25" width="450" height="420" fill="#D4A574" stroke="#000" strokeWidth="3"/>
          
          {/*Three-point line*/}
          <path d="M 25 95 Q 250 200 475 95" fill="none" stroke="#000" strokeWidth="2"/>
          
          {/*Free throw circle*/}
          <circle cx="250" cy="95" r="60" fill="none" stroke="#000" strokeWidth="2"/>
          
          {/*Key/Paint*/}
          <rect x="190" y="25" width="120" height="190" fill="rgba(255,255,255,0.1)" stroke="#000" strokeWidth="2"/>
          
          {/*Basketball hoop*/}
          <circle cx="250" cy="45" r="12" fill="none" stroke="#FF6600" strokeWidth="4"/>
          
          {/*Probability zones*/}
          {shotChart.map((zone, index) => (
            <g key={index}>
              <circle 
                cx={zone.x * 4.5 + 25} 
                cy={zone.y * 4.2 + 25} 
                r={zone.probability * 40}
                fill={zone.color}
                opacity="0.4"
                className="shot-zone"
              />
              <text 
                x={zone.x * 4.5 + 25} 
                y={zone.y * 4.2 + 30}
                textAnchor="middle"
                fill="#fff"
                fontSize="12"
                fontWeight="bold"
              >
                {(zone.probability * 100).toFixed(0)}%
              </text>
            </g>
          ))}
          
          {/*Player position*/}
          <circle cx="250" cy="200" r="8" fill="#00ff88" stroke="#fff" strokeWidth="2"/>
        </svg>
      </div>
      
      <div className="shot-zones-legend">
        {shotChart.map((zone, index) => (
          <div key={index} className="shot-zone-item">
            <div className="zone-color" style={{ backgroundColor: zone.color }}></div>
            <span>{zone.name}: {(zone.probability * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default InteractiveBasketballCourt;