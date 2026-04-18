import axios from "axios";
import type { VerifyNewsRequest, VerifyNewsResponse } from "./types";

const API_URL = "http://localhost:8000";

const api = axios.create({ baseURL: API_URL });

// Attach JWT to every request automatically
api.interceptors.request.use((config: any) => {
  const token = localStorage.getItem("token");
  if (token) {
    config.headers = config.headers || {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// On 401, clear token and reload (forces re-login)
api.interceptors.response.use(
  (response) => response,
  (error: any) => {
    if (error.response?.status === 401) {
      localStorage.removeItem("token");
      localStorage.removeItem("user");
      window.location.reload();
    }
    return Promise.reject(error);
  },
);

export default api;

// ── Typed helpers ────────────────────────────────────────────────────────────

export const verifyNews = async (req: VerifyNewsRequest) => {
  const { data } = await api.post<VerifyNewsResponse>("/sources/verify-news", req);
  return data;
};

