const BASE_URL = "http://localhost:8000"

export const fetchFullResults = (id) =>
  fetch(`${BASE_URL}/results/${id}`).then(r => r.json())

export const fetchRanking = (id) =>
  fetch(`${BASE_URL}/results/${id}/ranking`).then(r => r.json())

export const fetchTopTests = (id) =>
  fetch(`${BASE_URL}/results/${id}/top-tests`).then(r => r.json())

export const fetchMetrics = (id) =>
  fetch(`${BASE_URL}/results/${id}/metrics`).then(r => r.json())

export const fetchReduction = (id) =>
  fetch(`${BASE_URL}/results/${id}/reduction`).then(r => r.json())
