document.getElementById('predictForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const formData = new FormData(this);
    const result = document.getElementById('result');
    result.innerHTML = '<div class="alert alert-info">Calculando previsão...</div>';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (!data.success) {
            result.innerHTML = <div class="alert alert-danger">Erro: ${data.error}</div>;
            return;
        }

        const proba = (data.probability * 100).toFixed(2);
        const isRenew = data.prediction === 1;
        const alertClass = isRenew ? 'success' : 'danger';

        result.innerHTML = `
            <div class="alert alert-${alertClass}">
                <h4 class="alert-heading">${isRenew ? '✅ Cliente com alta chance de renovar' : '❌ Cliente com baixa chance de renovar'}</h4>
                <p class="mb-0"><strong>Probabilidade de renovação:</strong> ${proba}%</p>
            </div>
            <div id="gaugeChart"></div>
        `;

        Plotly.newPlot('gaugeChart', [{
            type: 'indicator',
            mode: 'gauge+number',
            value: parseFloat(proba),
            title: { text: 'Probabilidade de Renovação (%)' },
            gauge: {
                axis: { range: [0, 100] },
                bar: { color: isRenew ? '#2ecc71' : '#e74c3c' },
                steps: [
                    { range: [0, 50],  color: '#fadbd8' },
                    { range: [50, 100], color: '#d5f5e3' }
                ]
            }
        }], { height: 320 });

    } catch (err) {
        result.innerHTML = <div class="alert alert-danger">Erro inesperado: ${err.message}</div>;
    }
});