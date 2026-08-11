package main

import (
	"embed"
	"net/http"
)

//go:embed web/index.html web/app.js
var webAssets embed.FS

type embeddedWebRoute struct {
	path        string
	asset       string
	contentType string
}

var embeddedWebRoutes = []embeddedWebRoute{
	{path: "/", asset: "web/index.html", contentType: "text/html; charset=utf-8"},
	{path: "/app.js", asset: "web/app.js", contentType: "text/javascript; charset=utf-8"},
	{path: "/guide", asset: "web/index.html", contentType: "text/html; charset=utf-8"},
	{path: "/architecture", asset: "web/index.html", contentType: "text/html; charset=utf-8"},
}

func registerWebHandlers(mux *http.ServeMux) {
	for _, route := range embeddedWebRoutes {
		route := route
		mux.HandleFunc(route.path, func(w http.ResponseWriter, r *http.Request) {
			serveEmbeddedWebRoute(w, r, route)
		})
	}
}

func serveEmbeddedWebRoute(w http.ResponseWriter, r *http.Request, route embeddedWebRoute) {
	if r.URL.Path != route.path {
		http.NotFound(w, r)
		return
	}
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	body, err := webAssets.ReadFile(route.asset)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "failed to read asset")
		return
	}

	w.Header().Set("Content-Type", route.contentType)
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(body)
}
