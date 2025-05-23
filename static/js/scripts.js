

const choices = ['グー', 'チョキ', 'パー'];
const audio = new Audio('static/audio/janken.wav');
const bgm_title = new Audio('static/audio/bgm_cragy.mp3');
const bgm_game = new Audio('static/audio/bgm_kyaha.mp3');
let gameResults = [];
let maxCount = 20;
let GaugeTime = 1500;
let opponentID = generateOpponentID();
let wins = 0;
let loses = 0;
let draws = 0;
let points = [];
let next_choice = choices[Math.floor(Math.random() * choices.length)];
var socket = io();
let isGameRunning = false;
let min_entropy_prob = [0.,0.,0.];
let min_entropy_state = 0;

audio.volume = 1; // 音量を100%に設定
bgm_game.volume = 0.5; // 音量を50%に設定
bgm_game.loop = true; // ループ再生を有効にする
bgm_title.volume = 0.5; // 音量を50%に設定
bgm_title.loop = true; // ループ再生を有効にする

const sleep = (time) => new Promise((resolve) => setTimeout(resolve, time));//timeはミリ秒


const handImages = {
    "グー": "static/img/janken_gu.png",
    "チョキ": "static/img/janken_choki.png",
    "パー": "static/img/janken_pa.png",
    "Unknown": "static/img/mark_question.png",
};


let resultChart = null;
let probChart = null;

function init(){
    changeScreen("title-screen");
    document.getElementById("init-button").style.display = "none";
}

// ゲームの流れはここで実装
async function startGame() {
    changeScreen("game-screen");
    resetGame();
    isGameRunning = true;

    for(let i=0; i<maxCount; i++){
        if(!isGameRunning) {
            break;
        }
        document.getElementById("progress-bar").style.width = ((100*i)/(maxCount-1))+"%";
        document.getElementById("computer-hand").src = handImages["Unknown"];
        // audio.play();
        showText(["じゃん", "けん", "ぽん!"]);
        await sleep(GaugeTime);
        document.getElementById("computer-hand").src = handImages[next_choice];


        socket.emit("capture_hand");
        await new Promise((resolve) => {
            socket.once('capture_done', (response)=>{
                playerChoice = response.gesture;
                if(playerChoice === "Unknown"){
                    alert("手が認識できませんでした．もう一度やり直してください．");
                    i--;
                }else{
                    playGame(playerChoice);
                    console.log("win" + wins + " lose" + loses + " draw" + draws);
                }
                resolve();
            });
        });
        await sleep(1500);
    }

    if(!isGameRunning) {
        return;
    }
    socket.emit('save_data');
    await new Promise((resolve) => {
        socket.once('save_done', resolve);
    });
    socket.emit("calc_minentropy_state");
    await new Promise((resolve) => {
        socket.once('calc_minentropy_done', (response)=>{
            min_entropy_prob = response.prob;
            min_entropy_state = response.state;
            console.log("prob: " + response.prob);
            resolve();
        });
    });
    await update_score(wins, draws, loses);
    await sleep(500);
    let sum = wins + draws + loses;
    socket.emit("save_point", (wins - loses)/sum);
    changeScreen("result-screen");
    resetGame();
}

async function update_picture() {
    while(true) {
        socket.emit('update_picture');
        await new Promise((resolve) => {
            socket.once('updated_picture', (data) => {
                const blob = new Blob([data], { type: "image/jpeg" });
                const url = URL.createObjectURL(blob);
                const img = document.getElementById("cam-picture");
                img.src = url;
                // メモリ解放（必要に応じて）
                URL.revokeObjectURL(url);
                resolve();
            });
        });
        await sleep(200);
    }
}

// じゃんけんパート
function playGame(playerChoice) {

    let computerChoice = next_choice;
    let result = '';
    if (playerChoice === computerChoice) {
        result = '引き分け';
        draws += 1
    } else if (
        (playerChoice === 'グー' && computerChoice === 'チョキ') ||
        (playerChoice === 'チョキ' && computerChoice === 'パー') ||
        (playerChoice === 'パー' && computerChoice === 'グー')
    ) {
        result = '勝ち';
        wins += 1
    } else {
        result = '負け';
        loses += 1
    }
    
    // gameResults.push([opponentID, string2number(playerChoice), string2number(computerChoice), string2number(result)]);
    document.getElementById("wld").innerHTML = result;

    console.log("cpu: " + computerChoice);
    console.log("player: " + playerChoice);
    chooseNext(playerChoice, computerChoice);
    console.log("next_choice: " + next_choice);
}

function update_score(wins, draws, loses) {
    let sum = wins + loses + draws;
    if(wins > loses) {
        document.getElementById("result-message").innerHTML = "あなたの勝ち！";
        document.getElementById("result-message").style.color = "red";
    }
    else if(wins < loses) {
        document.getElementById("result-message").innerHTML = "CPUの勝ち！";
        document.getElementById("result-message").style.color = "blue";
    }
    else {
        document.getElementById("result-message").innerHTML = "引き分け！";
        document.getElementById("result-message").style.color = "black";
    }
}

function chooseNext(playerChoice, computerChoice) {
    socket.emit('choose', {var1:string2number(playerChoice), var2:string2number(computerChoice)}, 
        function(response) {
            next_choice = number2string(response);
    });
}

function generateOpponentID() {
    return Math.floor(Math.random() * 100000);
}

async function resetGame() {
    gameResults = [];
    draws = 0;
    wins = 0;
    loses = 0;
    document.getElementById("computer-hand").src = handImages["Unknown"];
    document.getElementById("wld").innerHTML = " ";
    socket.emit('reset');
    next_choice = choices[Math.floor(Math.random() * choices.length)];
    socket.emit("load_points");
    socket.once("load_points_done", (response) => {
        points = response.points;
    })
}

function string2number(str) {
    if (str === "グー") {
        return 0;
    } else if (str === "チョキ") {
        return 1;
    } else if (str === "パー") {
        return 2;
    }
}

function getHistogramData(points0, binCount=11) {
    const bins = Array(binCount).fill(0);
    const binWidth = 2.0 / binCount; // [-1.0, 1.0]の幅を11分割
    console.log("points0: ", points0);
    points0.forEach(p => {
        // [-1.0, 1.0]を[0, 2.0]にシフトしてビン番号を計算
        let idx = Math.floor((p + 1.0) / binWidth);
        if (idx < 0) idx = 0;
        if (idx >= binCount) idx = binCount - 1; // 1.0ちょうどは最後のビン
        bins[idx]++;
    });
    
    let sum = wins + draws + loses;
    let point = (wins - loses) / sum;
    let highlightIdx = Math.floor((point + 1.0) / binWidth);
    if (highlightIdx < 0) highlightIdx = 0;
    if (highlightIdx >= binCount) highlightIdx = binCount - 1;

    // 色配列を作成
    const colors = [];
    for(let i=0; i<binCount; i++){
        colors.push(i === highlightIdx ? 'red' : 'gray');
    }

    return [bins, colors];
}

function number2string(num) {
    if (num === 0) {
        return "グー";
    } else if (num === 1) {
        return "チョキ";
    } else if (num === 2) {
        return "パー";
    }
}

function sendResultsToSheet() {
    fetch("https://script.google.com/macros/s/AKfycbz1GBmKlrfGWxXjNBxE9XiRWZ6yR6ul0x92qaZJVJicBrdS_qxVRGTuIZ8gV6I7PaVL/exec", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        mode: "no-cors",
        body: JSON.stringify(gameResults)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => alert("データをスプレッドシートに送信しました！"))
    .catch(error => {
        console.error("fetch エラー:", error);
    });
}

function startGauge(){
    let progress = 0;
    document.getElementById("gauge-bar").style.width = "0%";
    document.getElementById("computer-hand").src = handImages["Unknown"];
    return new Promise((resolve) => {
        let interval = setInterval(function() {
            progress += 2;
            document.getElementById("gauge-bar").style.width = progress + "%";
            if (progress > 100) {
                document.getElementById("gauge-bar").style.width = "0%";
                clearInterval(interval);
                resolve();
            }
        }, GaugeTime / 50);
    });
}

function changeScreen(screenId){
    document.querySelectorAll('.screen').forEach(screen => {
        screen.classList.remove('active');
    });
    document.getElementById(screenId).classList.add('active');
    if(screenId === "result-screen"){
        if(!(probChart && resultChart)){
            graphsDefine();
        }
        graphUpdate();
    }    
    
    if(screenId === "title-screen"){
        bgm_title.play();
        bgm_game.pause();
        bgm_game.currentTime = 0; // 曲の先頭に戻す
    } else if(screenId === "game-screen"){
        bgm_title.pause();
        bgm_title.currentTime = 0; // 曲の先頭に戻す
        bgm_game.play();
    }else{
        bgm_title.pause();
        bgm_title.currentTime = 0; // 曲の先頭に戻す
        bgm_game.pause();
        bgm_game.currentTime = 0; // 曲の先頭に戻す
    }

    let sum = wins + draws + loses;
    let point = (wins - loses) / sum;
    socket.emit("get_top_percentile", point);
    socket.once("get_top_percentile_done", (response) => {
        const percentile = response.top_percentile;
        document.getElementById("percentile").innerHTML = `あなたのスコアは上位 <span style="color:red; font-size: 1.2em;">${percentile}%</span> です！`;
    });
    document.getElementById("pred-situ").innerHTML = num2state(min_entropy_state);
    let max_id = min_entropy_prob.indexOf(Math.max(...min_entropy_prob));
    document.getElementById("pred-result").innerHTML = "➡" + number2string(max_id) +"を出しやすい！";
}

function graphsDefine(){
    const ChartOptions = {
        responsive: true,
        animation: {
            duration: 1000,
            easing: 'easeOutQuad'
        },
        plugins: {
            legend: { display: false }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: { color: 'rgba(255,255,255,0.2)' },
                ticks: { color: '#fff' }
            },
            x: {
                grid: { display: false },
                ticks: { color: '#fff' }
            }
        }
    };

    const ctx = document.getElementById('result-graph').getContext('2d');
    resultChart = new Chart(ctx, {
        type: 'bar', // 棒グラフ
        data: {
            labels: ['勝ち', '負け', '引き分け'],
            datasets: [{
                label: 'じゃんけん結果',
                data: [0, 0, 0], // ここを動的に変更可能
                backgroundColor: [
                    'rgb(253, 159, 159)',
                    'rgb(163, 177, 248)',
                    'rgb(185, 185, 185)'
                ],
                borderRadius: 5, // 角丸
                borderSkipped: false
            }]
        },
        options: ChartOptions
    });

    const cty = document.getElementById('prob-graph').getContext('2d');
    probChart = new Chart(cty, {
        type: 'bar',
        data: {
            labels: ['グー', 'チョキ', 'パー'],
            datasets: [{
                label: '予測確率',
                data: [0, 0, 0],
                backgroundColor: [
                    'rgb(250, 199, 112)',
                    'rgb(252, 246, 126)',
                    'rgb(163, 243, 33)'
                ],
                borderRadius: 5, // 角丸
                borderSkipped: false
            }]
        },
        options: ChartOptions
    });

    const ctz = document.getElementById('pointdist-graph').getContext('2d');
    pointDistChart = new Chart(ctz, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: '分布',
                data: [],
                borderRadius: 5, // 角丸
                borderSkipped: false
            }]
        },
        options: ChartOptions
    });
}

function graphUpdate(){
    probChart.data.datasets[0].data = min_entropy_prob;
    probChart.update();
    resultChart.data.datasets[0].data = [wins, loses, draws];
    resultChart.update();
    const [histogramData, colors] = getHistogramData(points);
    pointDistChart.data.datasets[0].data = histogramData;
    pointDistChart.data.datasets[0].backgroundColor = colors;
    pointDistChart.data.labels = histogramData.map((_, i) => (i * 2.0 / histogramData.length - 1).toFixed(2));
    pointDistChart.update();
}

function num2state(num) {
    ans = "";
    if(Math.floor(num/3) == 0){
        ans = "グー";
    }else if(Math.floor(num/3) == 1){
        ans = "チョキ";
    }else if(Math.floor(num/3) == 2){
        ans = "パー";
    }
    ans += "を出して";
    if(num%3 == 0){
        ans += "負けた";
    }else if(num%3 == 1){
        ans += "引き分けた";
    }else if(num%3 == 2){
        ans += "勝った";
    }
    ans += "とき、次の手の確率は..."
    return ans;
}


function showText(words) {
    times = [500, 500, 800];
    const elem = document.getElementById("janken-text");
    elem.style.display = "block";
    for (let i = 0; i < words.length ; i++) {
        setTimeout(()=>{
            elem.textContent = words[i];
            elem.classList.add("popin-text");
            elem.style.opacity = 1;
            setTimeout(() => {
                elem.style.opacity = 0;
                elem.classList.remove("popin-text");
            }, times[i]);
        }, (i * GaugeTime) / (words.length - 1));
    }
}

document.addEventListener("keydown", function(event) {
    if (event.key === "Escape") {
        if(isGameRunning) {
            isGameRunning = false;
            alert("ゲームを中断しました．");
            changeScreen("title-screen");
            resetGame();
        }
    }
});