import React, { useState } from "react";
import api from "./api";
import type { User } from "./types";

type AuthPageProps = {
  onLogin: (user: User) => void;
};

function AuthPage({ onLogin }: AuthPageProps) {
  const [mode, setMode] = useState<string>("login");
  const [username, setUsername] = useState<string>("");
  const [email, setEmail] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      let tokenData: any;
      if (mode === "register") {
        const { data } = await api.post("/auth/register", {
          username,
          email,
          password,
        });
        tokenData = data;
      } else {
        const { data } = await api.post("/auth/login", {
          username,
          password,
        });
        tokenData = data;
      }

      localStorage.setItem("token", tokenData.access_token);
      const { data: user } = await api.get("/auth/me");
      localStorage.setItem("user", JSON.stringify(user));
      onLogin(user);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Помилка входу. Перевірте дані.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <h1>Disinformation Detector</h1>
        <div className="auth-tabs">
          <button
            type="button"
            className={mode === "login" ? "active" : ""}
            onClick={() => {
              setMode("login");
              setError(null);
            }}
          >
            Вхід
          </button>
          <button
            type="button"
            className={mode === "register" ? "active" : ""}
            onClick={() => {
              setMode("register");
              setError(null);
            }}
          >
            Реєстрація
          </button>
        </div>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            placeholder="Ім'я користувача"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
          {mode === "register" && (
            <input
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          )}
          <input
            type="password"
            placeholder="Пароль"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          {error && <div className="auth-error">{error}</div>}
          <button type="submit" disabled={loading}>
            {loading ? "Зачекайте..." : mode === "login" ? "Увійти" : "Зареєструватися"}
          </button>
        </form>
      </div>
    </div>
  );
}

export default AuthPage;

