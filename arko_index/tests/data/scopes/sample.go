// Scope fixture: a type, a method with a receiver and a function.
package pool

type Server struct {
	addr  string
	ready bool
}

func (s *Server) Start() error {
	s.ready = true
	return nil
}

func NewServer(addr string) *Server {
	return &Server{addr: addr}
}
