// Smooth scroll for nav links
document.querySelectorAll('.nav-links a').forEach(link => {
  link.addEventListener('click', e => {
    if (link.getAttribute('href').startsWith('#')) {
      e.preventDefault();
      document.querySelector(link.getAttribute('href')).scrollIntoView({
        behavior: 'smooth'
      });
    }
  });
});

// Button interactivity
document.getElementById("buyBtn").addEventListener("click", () => {
  alert("🍪 Thanks for your interest! Visit our store to buy fresh cookies!");
});

document.getElementById("contactBtn").addEventListener("click", () => {
  document.querySelector("#contact").scrollIntoView({ behavior: "smooth" });
});
