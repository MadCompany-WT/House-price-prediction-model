const form = document.querySelector("#predictForm");
const districtSelect = document.querySelector("#districtSelect");
const floorField = document.querySelector(".floor-field");
const landField = document.querySelector(".land-field");

const fields = {
    scenarioPrice: document.querySelector("#scenarioPrice"),
    objectSummary: document.querySelector("#objectSummary"),
    pricePerMeter: document.querySelector("#pricePerMeter"),
    deltaPrice: document.querySelector("#deltaPrice"),
    currentPrice: document.querySelector("#currentPrice"),
    infraMultiplier: document.querySelector("#infraMultiplier"),
    districtMultiplier: document.querySelector("#districtMultiplier"),
    landPrice: document.querySelector("#landPrice"),
    rawPrediction: document.querySelector("#rawPrediction"),
    materialMultiplier: document.querySelector("#materialMultiplier"),
    repairMultiplier: document.querySelector("#repairMultiplier"),
    floorMultiplier: document.querySelector("#floorMultiplier"),
    districtBars: document.querySelector("#districtBars"),
};

function formValue(name) {
    const input = form.elements[name];
    if (!input) return "";
    if (input instanceof RadioNodeList) {
        return input.value;
    }
    if (input.type === "checkbox") {
        return input.checked;
    }
    return input.value;
}

function collectPayload() {
    return {
        objectType: formValue("objectType"),
        district: formValue("district"),
        area: Number(formValue("area")),
        rooms: Number(formValue("rooms")),
        age: Number(formValue("age")),
        floor: Number(formValue("floor")),
        land: Number(formValue("land")),
        material: formValue("material"),
        repair: formValue("repair"),
        income: Number(formValue("income")),
        usdRate: Number(formValue("usdRate")),
        school: form.elements.school.checked,
        shop: form.elements.shop.checked,
        park: form.elements.park.checked,
    };
}

function updateRangeLabels() {
    document.querySelector("#ageValue").textContent = `${form.elements.age.value} лет`;
    document.querySelector("#landValue").textContent = `${form.elements.land.value} соток`;
    document.querySelector("#usdValue").textContent = `${form.elements.usdRate.value} ₸`;
}

function updateObjectMode() {
    const isHouse = formValue("objectType") === "house";
    floorField.classList.toggle("hidden", isHouse);
    landField.classList.toggle("hidden", !isHouse);
}

function renderDistrictOptions(districts, activeDistrict) {
    const currentValues = Array.from(districtSelect.options).map((option) => option.value).join("|");
    const nextValues = districts.map((district) => district.name).join("|");

    if (currentValues === nextValues) {
        districtSelect.value = activeDistrict;
        return;
    }

    districtSelect.innerHTML = "";
    districts.forEach((district) => {
        const option = document.createElement("option");
        option.value = district.name;
        option.textContent = district.name;
        districtSelect.append(option);
    });
    districtSelect.value = activeDistrict;
}

function renderDistrictBars(districts) {
    const maxPrice = Math.max(...districts.map((district) => district.price), 1);
    fields.districtBars.innerHTML = "";

    districts.forEach((district) => {
        const row = document.createElement("article");
        row.className = "district-row";
        row.innerHTML = `
            <div class="district-name">
                <strong>${district.name}</strong>
                <span>${district.note}</span>
            </div>
            <div class="bar-track" aria-hidden="true">
                <div class="bar-fill" style="--bar-width: ${(district.price / maxPrice) * 100}%"></div>
            </div>
            <div class="district-price">${district.priceText}</div>
        `;
        fields.districtBars.append(row);
    });
}

function renderResult(data) {
    const objectText = data.params.objectType === "house" ? "частный дом" : "квартира";
    renderDistrictOptions(data.availableDistricts, data.params.district);

    fields.scenarioPrice.textContent = data.scenarioPriceText;
    fields.objectSummary.textContent = `${data.params.district}, ${objectText}, ${data.params.area} м², ${data.params.rooms} комн.`;
    fields.pricePerMeter.textContent = `${data.pricePerMeterText} за м²`;
    fields.deltaPrice.textContent = `${data.deltaText} от курса`;
    fields.currentPrice.textContent = data.currentPriceText;
    fields.infraMultiplier.textContent = `x${data.infraMultiplier.toFixed(2)}`;
    fields.districtMultiplier.textContent = `x${data.districtMultiplier.toFixed(2)}`;
    fields.landPrice.textContent = data.landPriceText;
    fields.rawPrediction.textContent = data.rawPrediction;
    fields.materialMultiplier.textContent = `x${data.materialMultiplier.toFixed(2)}`;
    fields.repairMultiplier.textContent = `x${data.repairMultiplier.toFixed(2)}`;
    fields.floorMultiplier.textContent = `x${data.floorMultiplier.toFixed(2)}`;
    renderDistrictBars(data.districts);
}

async function predict() {
    updateRangeLabels();
    updateObjectMode();

    const response = await fetch("/api/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(collectPayload()),
    });

    if (!response.ok) {
        fields.scenarioPrice.textContent = "Ошибка расчета";
        return;
    }

    renderResult(await response.json());
}

form.addEventListener("input", predict);
form.addEventListener("change", predict);
predict();
