import React from 'react';
import { Activity, Brain, Timer, Target, Zap, Clock } from 'lucide-react';

const GameForm = ({ formData, handleChange, handleSubmit, loading }) => {
  return (
    <div className="glass-card">
      <div className="game-form">
        <div className="form-header">
          <Brain className="form-icon" />
          <h2>Advanced Game Analysis</h2>
        </div>
        
        <div className="form-grid">
          <div className="form-group">
            <label className="form-label">
              <Activity size={16} />
              Sport
            </label>
            <select name="sport" value={formData.sport} onChange={handleChange} className="form-input">
              <option value="football">🏈 Football</option>
              <option value="basketball">🏀 Basketball</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">
              <Timer size={16} />
              Quarter
            </label>
            <select name="quarter" value={formData.quarter} onChange={handleChange} className="form-input">
              <option value="1">1st Quarter</option>
              <option value="2">2nd Quarter</option>
              <option value="3">3rd Quarter</option>
              <option value="4">4th Quarter</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">
              <Timer size={16} />
              Game Time
            </label>
            <div className="time-inputs">
              <input 
                type="number" 
                name="time_minutes" 
                placeholder="Min" 
                value={formData.time_minutes} 
                onChange={handleChange} 
                className="form-input time-input"
                min="0" 
                max="15"
              />
              <span className="time-separator">:</span>
              <input 
                type="number" 
                name="time_seconds" 
                placeholder="Sec" 
                value={formData.time_seconds} 
                onChange={handleChange} 
                className="form-input time-input"
                min="0" 
                max="59"
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">
              <Target size={16} />
              Score (Home vs Away)
            </label>
            <div className="score-inputs">
              <input 
                type="number" 
                name="score_home" 
                placeholder="Home" 
                value={formData.score_home} 
                onChange={handleChange} 
                className="form-input score-input"
                min="0"
              />
              <span className="score-separator">-</span>
              <input 
                type="number" 
                name="score_away" 
                placeholder="Away" 
                value={formData.score_away} 
                onChange={handleChange} 
                className="form-input score-input"
                min="0"
              />
            </div>
          </div>

          {formData.sport === 'football' && (
            <>
              <div className="form-group">
                <label className="form-label">Down & Distance</label>
                <div className="down-distance">
                  <select name="down" value={formData.down} onChange={handleChange} className="form-input">
                    <option value="1">1st Down</option>
                    <option value="2">2nd Down</option>
                    <option value="3">3rd Down</option>
                    <option value="4">4th Down</option>
                  </select>
                  <input 
                    type="number" 
                    name="distance" 
                    placeholder="Yards to go" 
                    value={formData.distance} 
                    onChange={handleChange} 
                    className="form-input"
                    min="1" 
                    required 
                  />
                </div>
              </div>
              
              <div className="form-group">
                <label className="form-label">Field Position</label>
                <div className="yard-line-container">
                  <input 
                    type="range"
                    name="yard_line" 
                    value={formData.yard_line || 50} 
                    onChange={handleChange} 
                    className="yard-line-slider"
                    min="1" 
                    max="99"
                  />
                  <span className="yard-line-value">{formData.yard_line || 50} Yard Line</span>
                </div>
              </div>
            </>
          )}

          {formData.sport === 'basketball' && (
            <div className="form-group">
              <label className="form-label">
                <Clock size={16} />
                Shot Clock
              </label>
              <input 
                type="number" 
                name="shot_clock" 
                placeholder="Seconds" 
                value={formData.shot_clock} 
                onChange={handleChange} 
                className="form-input"
                min="0"
                max="24"
              />
            </div>
          )}
        </div>

        <button 
          type="button" 
          className={`predict-button ${loading ? 'loading' : ''}`}
          disabled={loading}
          onClick={handleSubmit}
        >
          {loading ? (
            <>
              <div className="spinner"></div>
              Analyzing with AI...
            </>
          ) : (
            <>
              <Zap size={20} />
              Run Advanced Prediction
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default GameForm;