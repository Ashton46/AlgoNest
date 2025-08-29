import React from 'react';

const GameForm = ({ formData, handleChange, handleSubmit }) => {
  return (
    <form className="game-form" onSubmit={handleSubmit}>
      <h2>Game Situation</h2>
      <label>
        Sport:
        <select name="sport" value={formData.sport} onChange={handleChange}>
          <option value="football">Football</option>
          <option value="basketball">Basketball</option>
        </select>
      </label>

      {formData.sport === 'football' && (
        <>
          <label>
            Down:
            <select name="down" value={formData.down} onChange={handleChange}>
              <option value="1">1st</option>
              <option value="2">2nd</option>
              <option value="3">3rd</option>
              <option value="4">4th</option>
            </select>
          </label>
          <label>
            Distance to go:
            <input type="number" name="distance" value={formData.distance} onChange={handleChange} min="1" required />
          </label>
          <label>
            Yard Line:
            <input type="number" name="yard_line" value={formData.yard_line} onChange={handleChange} min="1" max="100" required />
          </label>
        </>
      )}

      <button type="submit">Predict Play</button>
    </form>
  );
};

export default GameForm;