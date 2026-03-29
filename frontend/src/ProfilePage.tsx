import React from "react";

type ProfilePageProps = {
  user: any;
};

function ProfilePage({ user }: ProfilePageProps) {
  const formatDate = (iso: any) =>
    new Date(iso).toLocaleDateString("uk-UA", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });

  return (
    <div className="section-block">
      <h2>Профіль користувача</h2>
      <div className="profile-card">
        <div className="profile-avatar">
          {user.username.charAt(0).toUpperCase()}
        </div>
        <table className="profile-table">
          <tbody>
            <tr>
              <th>Ім'я користувача</th>
              <td>{user.username}</td>
            </tr>
            <tr>
              <th>Email</th>
              <td>{user.email}</td>
            </tr>
            <tr>
              <th>Дата реєстрації</th>
              <td>{formatDate(user.created_at)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default ProfilePage;

