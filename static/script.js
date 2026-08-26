/* ===================================================
   Gender Classifier — Frontend Logic
   =================================================== */
(function () {
    "use strict";

    // DOM refs
    const form        = document.getElementById("predict-form");
    const nameInput   = document.getElementById("name-input");
    const predictBtn  = document.getElementById("predict-btn");
    const resultCard  = document.getElementById("result-card");
    const resultName  = document.getElementById("result-name");
    const resultGender = document.getElementById("result-gender");
    const iconMale    = document.getElementById("icon-male");
    const iconFemale  = document.getElementById("icon-female");
    const iconWrapper = document.getElementById("gender-icon-wrapper");
    const confValue   = document.getElementById("confidence-value");
    const confBar     = document.getElementById("confidence-bar");
    const errorToast  = document.getElementById("error-toast");
    const errorMsg    = document.getElementById("error-msg");

    // ---------- helpers ----------
    function setLoading(on) {
        predictBtn.classList.toggle("loading", on);
        predictBtn.disabled = on;
    }

    function showError(msg) {
        errorMsg.textContent = msg;
        errorToast.classList.remove("hidden");
        // Force reflow so transition fires
        void errorToast.offsetWidth;
        errorToast.classList.add("show");
        setTimeout(() => {
            errorToast.classList.remove("show");
            setTimeout(() => errorToast.classList.add("hidden"), 350);
        }, 4000);
    }

    function animateConfidence(target) {
        confBar.style.width = "0%";
        confValue.textContent = "0%";

        // Small delay so CSS transition kicks in
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                confBar.style.width = target + "%";
            });
        });

        // Count-up number
        const duration = 800;
        const start = performance.now();
        function step(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            const current = (progress * target).toFixed(1);
            confValue.textContent = current + "%";
            if (progress < 1) requestAnimationFrame(step);
            else confValue.textContent = target + "%";
        }
        requestAnimationFrame(step);
    }

    function showResult(data) {
        const isMale = data.gender_code === "m";

        // Name
        resultName.textContent = data.name;

        // Gender label
        resultGender.textContent = data.gender;
        resultGender.className = "result-gender " + (isMale ? "male" : "female");

        // Icon
        iconMale.classList.toggle("hidden", !isMale);
        iconFemale.classList.toggle("hidden", isMale);
        iconWrapper.className = "gender-icon-wrapper " + (isMale ? "male-bg" : "female-bg");

        // Confidence bar colour
        confBar.className = "confidence-bar " + (isMale ? "male" : "female");

        // Show card
        resultCard.classList.remove("hidden");
        // Re-trigger animation
        resultCard.style.animation = "none";
        void resultCard.offsetWidth;
        resultCard.style.animation = "";

        // Animate confidence
        animateConfidence(data.confidence);
    }

    // ---------- form submit ----------
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const name = nameInput.value.trim();
        if (!name) return;

        setLoading(true);

        try {
            const res = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name }),
            });

            if (!res.ok) {
                const err = await res.json().catch(() => null);
                throw new Error(err?.error || "Server returned " + res.status);
            }

            const data = await res.json();
            showResult(data);
        } catch (err) {
            showError(err.message || "Could not reach the server.");
        } finally {
            setLoading(false);
        }
    });

    // Focus input on load
    nameInput.focus();
})();
