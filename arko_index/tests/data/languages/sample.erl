%%% A key-value store server.
-module(kv_server).
-behaviour(gen_server).

-export([start_link/0, put/2, get/1]).
-export([init/1, handle_call/3, handle_cast/2]).

-record(state, {table = #{} :: map(), hits = 0 :: integer()}).

-define(SERVER, ?MODULE).

-type key() :: atom() | binary().

%% @doc Starts the server.
-spec start_link() -> {ok, pid()}.
start_link() ->
    gen_server:start_link({local, ?SERVER}, ?MODULE, [], []).

%% @doc Stores a value.
put(Key, Value) ->
    gen_server:call(?SERVER, {put, Key, Value}).

get(Key) ->
    gen_server:call(?SERVER, {get, Key}).

init([]) ->
    {ok, #state{}}.

handle_call({put, Key, Value}, _From, State = #state{table = T}) ->
    {reply, ok, State#state{table = T#{Key => Value}}};
handle_call({get, Key}, _From, State = #state{table = T, hits = H}) ->
    case maps:find(Key, T) of
        {ok, V} -> {reply, {ok, V}, State#state{hits = H + 1}};
        error -> {reply, not_found, State}
    end.

handle_cast(_Msg, State) ->
    {noreply, State}.
