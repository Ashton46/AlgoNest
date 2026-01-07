import React, { useState, useEffect, useRef } from 'react';
import { Map, TrendingUp } from 'lucide-react';
import * as THREE from 'three';
import { generateHeatmapData } from '../utils/dataGenerators';

const InteractiveFootballField = ({ prediction, formData, animateField }) => {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const animationRef = useRef(null);
  const [selectedZone, setSelectedZone] = useState(null);

  const getMatchingPrediction = (predictions, zoneName) => {
    const searchTerms = zoneName.toLowerCase().split(' ');
    return predictions.find(p => 
      p.play_type && searchTerms.some(term => 
        p.play_type.toLowerCase().includes(term)
      )
    );
  };

  const getZoneProbability = (predictions, zoneName) => {
    const directMatch = getMatchingPrediction(predictions, zoneName);
    if (directMatch) {
      return directMatch.probability;
    }

    const runPrediction = predictions.find(p =>
      p.play_type && p.play_type.toLowerCase().includes('run')
    );
    const passPrediction = predictions.find(p =>
      p.play_type && p.play_type.toLowerCase().includes('pass')
    );

    if (runPrediction) {
      if (zoneName === 'Run Left' || zoneName === 'Run Center' || zoneName === 'Run Right') {
        return runPrediction.probability / 3;
      }
    }

    if (passPrediction) {
      if (zoneName === 'Pass Short Left' || zoneName === 'Pass Short Right') {
        return passPrediction.probability * 0.35;
      }
      if (zoneName === 'Pass Deep') {
        return passPrediction.probability * 0.3;
      }
    }

    return 0;
  };

  const normalizeZoneProbabilities = (zones) => {
    const total = zones.reduce((sum, zone) => sum + (zone.adjustedProbability || 0), 0);
    if (total <= 0) {
      return zones;
    }
    return zones.map(zone => ({
      ...zone,
      adjustedProbability: zone.adjustedProbability / total
    }));
  };

  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    
    const camera = new THREE.PerspectiveCamera(75, 400 / 300, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(400, 300);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    
    mountRef.current.appendChild(renderer.domElement);

    // Field geometry
    const fieldGeometry = new THREE.PlaneGeometry(20, 10);
    const fieldMaterial = new THREE.MeshLambertMaterial({ 
      color: 0x2d5016,
      transparent: true,
      opacity: 0.8
    });
    const field = new THREE.Mesh(fieldGeometry, fieldMaterial);
    field.rotation.x = -Math.PI / 2;
    field.receiveShadow = true;
    scene.add(field);

    // Yard lines
    for (let i = 0; i <= 10; i++) {
      const lineGeometry = new THREE.PlaneGeometry(0.1, 10);
      const lineMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
      const line = new THREE.Mesh(lineGeometry, lineMaterial);
      line.position.set(-10 + (i * 2), 0.01, 0);
      line.rotation.x = -Math.PI / 2;
      scene.add(line);
    }

    // Goal posts
    const goalGeometry = new THREE.CylinderGeometry(0.1, 0.1, 3);
    const goalMaterial = new THREE.MeshLambertMaterial({ color: 0xffff00 });
    
    const goalPost1 = new THREE.Mesh(goalGeometry, goalMaterial);
    goalPost1.position.set(-10, 1.5, -2);
    scene.add(goalPost1);
    
    const goalPost2 = new THREE.Mesh(goalGeometry, goalMaterial);
    goalPost2.position.set(-10, 1.5, 2);
    scene.add(goalPost2);

    // Football
    const ballGeometry = new THREE.SphereGeometry(0.2, 8, 6);
    const ballMaterial = new THREE.MeshLambertMaterial({ color: 0x8B4513 });
    const ball = new THREE.Mesh(ballGeometry, ballMaterial);
    
    const yardLine = parseInt(formData.yard_line) || 50;
    const ballPosition = ((100 - yardLine) / 100) * 20 - 10;
    ball.position.set(ballPosition, 0.3, 0);
    ball.castShadow = true;
    scene.add(ball);

    if (prediction && prediction.predictions) {
      const heatmapData = generateHeatmapData(yardLine);
      
      heatmapData.forEach((zone) => {
        const probability = getZoneProbability(prediction.predictions, zone.name);
        
        if (probability > 0) {
          const zoneGeometry = new THREE.SphereGeometry(probability * 3, 16, 16);
          const zoneMaterial = new THREE.MeshLambertMaterial({ 
            color: new THREE.Color(zone.color),
            transparent: true,
            opacity: 0.3 + (probability * 0.7)
          });
          const zoneMesh = new THREE.Mesh(zoneGeometry, zoneMaterial);
          
          const x = (zone.x - 0.5) * 20;
          const z = (zone.y - 0.5) * 10;
          zoneMesh.position.set(x, probability * 2, z);
          
          scene.add(zoneMesh);
        }
      });
    }

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    scene.add(directionalLight);

    camera.position.set(0, 8, 12);
    camera.lookAt(0, 0, 0);

    let animationSpeed = 0;
    const animate = () => {
      animationRef.current = requestAnimationFrame(animate);
      
      if (animateField) {
        animationSpeed += 0.05;
        ball.position.y = 0.3 + Math.sin(animationSpeed) * 0.2;
        camera.position.x = Math.sin(animationSpeed * 0.5) * 2;
      }
      
      renderer.render(scene, camera);
    };
    
    animate();
    sceneRef.current = { scene, camera, renderer, ball };

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [formData, animateField, prediction]);

  const heatmapData = generateHeatmapData(parseInt(formData.yard_line) || 50);

  const updatedHeatmapData = normalizeZoneProbabilities(heatmapData.map(zone => {
    if (prediction && prediction.predictions) {
      return {
        ...zone,
        adjustedProbability: getZoneProbability(prediction.predictions, zone.name)
      };
    }
    return {
      ...zone,
      adjustedProbability: 0
    };
  }));

  return (
    <div className="interactive-field-container">
      <div className="field-header">
        <Map size={24} />
        <h3>3D Field Analysis</h3>
        <div className="field-stats">
          <span>Yard Line: {formData.yard_line || 50}</span>
          <span>Down: {formData.down}</span>
        </div>
      </div>
      
      <div className="field-visualization-grid">
        <div className="three-d-field" ref={mountRef}></div>
        
        <div className="heatmap-legend">
          <h4>Probability Heatmap</h4>
          <div className="legend-items">
            {updatedHeatmapData.map((zone) => (
              <div 
                key={zone.id} 
                className={`legend-item ${selectedZone === zone.id ? 'selected' : ''}`}
                onClick={() => setSelectedZone(selectedZone === zone.id ? null : zone.id)}
              >
                <div 
                  className="color-indicator" 
                  style={{ backgroundColor: zone.color }}
                ></div>
                <span className="zone-name">{zone.name}</span>
                <span className="zone-probability">
                  {zone.adjustedProbability > 0 
                    ? `${(zone.adjustedProbability * 100).toFixed(1)}%`
                    : 'No data'
                  }
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      <div className="field-controls">
        <div className="control-group">
          <label>Field View:</label>
          <select className="control-select">
            <option>3D Perspective</option>
            <option>Top Down</option>
            <option>Side View</option>
          </select>
        </div>
        <div className="control-group">
          <label>Show:</label>
          <div className="checkbox-group">
            <label><input type="checkbox" defaultChecked /> Heatmap</label>
            <label><input type="checkbox" defaultChecked /> Yard Lines</label>
            <label><input type="checkbox" /> Player Positions</label>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InteractiveFootballField;