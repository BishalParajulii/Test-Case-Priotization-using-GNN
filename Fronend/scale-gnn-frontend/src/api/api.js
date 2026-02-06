export const API_BASE = 'http://localhost:8000';

export const fetchJSON = async (url, options = {}) => {
  const res = await fetch(`${API_BASE}${url}`, options);
  if (!res.ok) throw new Error(`API Error: ${res.status}`);
  return res.json();
};
