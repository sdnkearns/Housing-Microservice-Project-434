package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/gorilla/mux"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"gonum.org/v1/gonum/mat"
)

// Prometheus

var (
	httpRequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "http_requests_total",
			Help: "Total HTTP requests, labeled by route, method, and status code.",
		},
		[]string{"route", "method", "status"},
	)

	httpRequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "http_request_duration_seconds",
			Help:    "HTTP request latency in seconds, labeled by route and method.",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"route", "method"},
	)

	predictionsTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "predictions_total",
			Help: "Total number of predictions served by /predict.",
		},
	)

	modelR2 = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "model_r2",
			Help: "R-squared of the currently loaded linear regression model on its training data.",
		},
	)

	modelTrainingSamples = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "model_training_samples",
			Help: "Number of rows used to train the current model.",
		},
	)
)

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(status int) {
	r.status = status
	r.ResponseWriter.WriteHeader(status)
}

func metricsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}

		next.ServeHTTP(rec, r)

		route := "unmatched"
		if m := mux.CurrentRoute(r); m != nil {
			if tmpl, err := m.GetPathTemplate(); err == nil {
				route = tmpl
			}
		}
		duration := time.Since(start).Seconds()
		httpRequestsTotal.WithLabelValues(route, r.Method, strconv.Itoa(rec.status)).Inc()
		httpRequestDuration.WithLabelValues(route, r.Method).Observe(duration)
	})
}

var featureNames = []string{
	"OverallQual",
	"OverallCond",
	"GrLivArea",
	"LotArea",
	"YearBuilt",
	"YearRemodAdd",
	"TotalBsmtSF",
	"BsmtFinSF1",
	"GarageCars",
	"GarageArea",
	"FullBath",
	"HalfBath",
	"BedroomAbvGr",
	"TotRmsAbvGrd",
	"Fireplaces",
	"WoodDeckSF",
	"OpenPorchSF",
	"MasVnrArea",
}

type House struct {
	ID        int                `json:"id"`
	Features  map[string]float64 `json:"features"`
	SalePrice float64            `json:"salePrice,omitempty"`
}

type LinearModel struct {
	Coefficients []float64
	FeatureNames []string
	R2           float64
	TrainedAt    time.Time
	NumSamples   int
}

func featureVector(h House) []float64 {
	fv := make([]float64, len(featureNames))
	for i, name := range featureNames {
		fv[i] = h.Features[name]
	}
	return fv
}

func trainLinearModel(houses []House) (*LinearModel, error) {
	n := len(houses)
	if n == 0 {
		return nil, fmt.Errorf("no training rows supplied")
	}
	p := len(featureVector(houses[0])) + 1 // +1 for intercept column

	xData := make([]float64, n*p)
	yData := make([]float64, n)

	for i, h := range houses {
		xData[i*p] = 1.0
		fv := featureVector(h)
		for j, v := range fv {
			xData[i*p+j+1] = v
		}
		yData[i] = h.SalePrice
	}

	X := mat.NewDense(n, p, xData)
	y := mat.NewVecDense(n, yData)

	var xtx mat.Dense
	xtx.Mul(X.T(), X)

	var xtxInv mat.Dense
	if err := xtxInv.Inverse(&xtx); err != nil {
		return nil, fmt.Errorf("could not invert X^T X (check for collinear features): %w", err)
	}

	var xty mat.VecDense
	xty.MulVec(X.T(), y)

	var beta mat.VecDense
	beta.MulVec(&xtxInv, &xty)

	coeffs := make([]float64, p)
	for i := 0; i < p; i++ {
		coeffs[i] = beta.AtVec(i)
	}

	model := &LinearModel{
		Coefficients: coeffs,
		FeatureNames: append([]string{"intercept"}, featureNames...),
		TrainedAt:    time.Now(),
		NumSamples:   n,
	}
	model.R2 = computeR2(model, houses)
	return model, nil
}

func (m *LinearModel) Predict(h House) float64 {
	pred := m.Coefficients[0]
	for i, v := range featureVector(h) {
		pred += m.Coefficients[i+1] * v
	}
	return pred
}

func computeR2(m *LinearModel, houses []House) float64 {
	var mean, ssTot, ssRes float64
	for _, h := range houses {
		mean += h.SalePrice
	}
	mean /= float64(len(houses))

	for _, h := range houses {
		pred := m.Predict(h)
		ssRes += (h.SalePrice - pred) * (h.SalePrice - pred)
		ssTot += (h.SalePrice - mean) * (h.SalePrice - mean)
	}
	if ssTot == 0 {
		return 0
	}
	return 1 - ssRes/ssTot
}

func loadHousesCSV(path string, requireSalePrice bool) ([]House, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.FieldsPerRecord = -1

	header, err := r.Read()
	if err != nil {
		return nil, err
	}
	col := map[string]int{}
	for i, name := range header {
		col[name] = i
	}

	required := append([]string{"Id"}, featureNames...)
	if requireSalePrice {
		required = append(required, "SalePrice")
	}
	for _, name := range required {
		if _, ok := col[name]; !ok {
			return nil, fmt.Errorf("expected column %q not found in %s", name, path)
		}
	}

	get := func(row []string, name string) (float64, bool) {
		idx, ok := col[name]
		if !ok || idx >= len(row) || row[idx] == "" || row[idx] == "NA" {
			return 0, false
		}
		v, err := strconv.ParseFloat(row[idx], 64)
		return v, err == nil
	}

	var houses []House
	for {
		row, err := r.Read()
		if err != nil {
			break
		}
		id, _ := strconv.Atoi(row[col["Id"]])

		features := make(map[string]float64, len(featureNames))
		rowOK := true
		for _, name := range featureNames {
			v, ok := get(row, name)
			if !ok {
				rowOK = false
				break
			}
			features[name] = v
		}
		if !rowOK {
			continue
		}

		h := House{ID: id, Features: features}
		if requireSalePrice {
			sp, ok := get(row, "SalePrice")
			if !ok {
				continue
			}
			h.SalePrice = sp
		}
		houses = append(houses, h)
	}
	return houses, nil
}

type Server struct {
	houses []House // in-memory "resource" store, loaded from train.csv
	model  *LinearModel
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func (s *Server) listResources(w http.ResponseWriter, r *http.Request) {
	limit := 25
	if q := r.URL.Query().Get("limit"); q != "" {
		if n, err := strconv.Atoi(q); err == nil {
			limit = n
		}
	}
	if limit > len(s.houses) {
		limit = len(s.houses)
	}
	writeJSON(w, http.StatusOK, s.houses[:limit])
}

func (s *Server) getResource(w http.ResponseWriter, r *http.Request) {
	id, _ := strconv.Atoi(mux.Vars(r)["id"])
	for _, h := range s.houses {
		if h.ID == id {
			writeJSON(w, http.StatusOK, h)
			return
		}
	}
	writeJSON(w, http.StatusNotFound, map[string]string{"error": "resource not found"})
}

func (s *Server) getResourceValues(w http.ResponseWriter, r *http.Request) {
	id, _ := strconv.Atoi(mux.Vars(r)["id"])
	for _, h := range s.houses {
		if h.ID == id {
			writeJSON(w, http.StatusOK, featureVector(h))
			return
		}
	}
	writeJSON(w, http.StatusNotFound, map[string]string{"error": "resource not found"})
}

func (s *Server) predict(w http.ResponseWriter, r *http.Request) {
	if s.model == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "model not trained"})
		return
	}
	var h House
	if err := json.NewDecoder(r.Body).Decode(&h); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON body: " + err.Error()})
		return
	}
	pred := s.model.Predict(h)
	predictionsTotal.Inc()
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"input":              h,
		"predictedSalePrice": pred,
	})
}

func (s *Server) modelInfo(w http.ResponseWriter, r *http.Request) {
	if s.model == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "model not trained"})
		return
	}
	writeJSON(w, http.StatusOK, s.model)
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func main() {
	dataPath := os.Getenv("TRAIN_CSV")
	if dataPath == "" {
		dataPath = "data/train.csv"
	}

	houses, err := loadHousesCSV(dataPath, true)
	if err != nil {
		log.Fatalf("failed to load training data from %s: %v", dataPath, err)
	}
	log.Printf("loaded %d training rows from %s", len(houses), dataPath)

	model, err := trainLinearModel(houses)
	if err != nil {
		log.Fatalf("failed to train model: %v", err)
	}
	log.Printf("trained linear regression model on %d samples, R^2=%.4f", model.NumSamples, model.R2)
	modelR2.Set(model.R2)
	modelTrainingSamples.Set(float64(model.NumSamples))

	srv := &Server{houses: houses, model: model}

	router := mux.NewRouter()
	router.Use(metricsMiddleware)
	router.HandleFunc("/health", srv.health).Methods("GET")
	router.HandleFunc("/resources", srv.listResources).Methods("GET")
	router.HandleFunc("/resources/{id:[0-9]+}", srv.getResource).Methods("GET")
	router.HandleFunc("/resources/{id:[0-9]+}/values", srv.getResourceValues).Methods("GET")
	router.HandleFunc("/predict", srv.predict).Methods("POST")
	router.HandleFunc("/model/info", srv.modelInfo).Methods("GET")
	router.Handle("/metrics", promhttp.Handler()).Methods("GET")

	httpSrv := &http.Server{
		Handler:      router,
		Addr:         ":8080",
		WriteTimeout: 15 * time.Second,
		ReadTimeout:  15 * time.Second,
	}
	log.Println("listening on :8080")
	log.Fatal(httpSrv.ListenAndServe())
}
