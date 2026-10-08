module Counter exposing (Model, Msg(..), init, update, view)

{-| A counter with a step size.
-}

import Html exposing (Html, button, div, text)
import Html.Events exposing (onClick)


type alias Model =
    { count : Int
    , step : Int
    }


type Msg
    = Increment
    | Decrement
    | SetStep Int


{-| The starting model.
-}
init : Model
init =
    { count = 0, step = 1 }


update : Msg -> Model -> Model
update msg model =
    case msg of
        Increment ->
            { model | count = model.count + model.step }

        Decrement ->
            { model | count = model.count - model.step }

        SetStep step ->
            { model | step = step }


-- The view: two buttons and the count.
view : Model -> Html Msg
view model =
    div []
        [ button [ onClick Decrement ] [ text "-" ]
        , text (String.fromInt model.count)
        , button [ onClick Increment ] [ text "+" ]
        ]


port save : Int -> Cmd msg
