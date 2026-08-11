package main

import (
	"errors"
	"net/http"

	"awesomeProject/internal/control"
)

type controlDispatchConfirmer interface {
	ConfirmDispatch(commandID string) (control.DispatchConfirmation, error)
}

func registerControlDispatchHandlers(mux *http.ServeMux, commands controlDispatchConfirmer) {
	mux.HandleFunc("/control/dispatch/confirm", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}

		var request struct {
			CommandID string `json:"commandId"`
		}
		if err := decodeJSONBody(r, &request); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}

		confirmation, err := commands.ConfirmDispatch(request.CommandID)
		if err != nil {
			if errors.Is(err, control.ErrDispatchCommandIDRequired) {
				writeError(w, http.StatusBadRequest, err.Error())
				return
			}
			writeError(w, http.StatusInternalServerError, err.Error())
			return
		}

		writeJSON(w, http.StatusOK, confirmation)
	})
}
