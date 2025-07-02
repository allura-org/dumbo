from typing import Callable, TypeVar, TypeGuard, Awaitable, Generic

T = TypeVar('T')
U = TypeVar('U')

# internal class used to allow representing Result[None]
class FakeNone:
    def __bool__(self) -> bool:
        return False

class Result(Generic[T]):
    _value: T | FakeNone
    _error: BaseException | FakeNone

    def __init__(self, value: T | FakeNone, error: BaseException | FakeNone):
        self._value = value
        self._error = error
    
    def unwrap(self) -> T:
        if isinstance(self._error, BaseException) or isinstance(self._value, FakeNone):
            if isinstance(self._error, BaseException):
                raise self._error
            else:
                raise Exception(self._error)
        return self._value

    def unwrap_err(self) -> BaseException:
        if isinstance(self._error, FakeNone):
            raise Exception("Result is ok")
        return self._error
    
    def unwrap_or(self, default: T) -> T:
        if isinstance(self._error, BaseException) or isinstance(self._value, FakeNone):
            return default
        return self._value

    def map_ok(self, func: Callable[[T], U]) -> "Result[U]":
        if isinstance(self._error, BaseException) or isinstance(self._value, FakeNone):
            return Result(None, self._error)
        return Result(func(self._value), self._error)

    @staticmethod
    def resultify(func: Callable[..., T]) -> "Callable[..., Result[T]]":
        def wrapper(*args, **kwargs) -> Result[T]:
            try:
                return Result(func(*args, **kwargs), FakeNone())
            except Exception as e:
                return Result(FakeNone(), e)
        return wrapper

    @staticmethod
    def resultify_async(func: Callable[..., Awaitable[T]]) -> "Callable[..., Awaitable[Result[T]]]":
        async def wrapper(*args, **kwargs) -> Result[T]:
            try:
                result = await func(*args, **kwargs)
                return Result(result, FakeNone())
            except Exception as e:
                return Result(FakeNone(), e)
        return wrapper

# External TypeGuard functions that actually work
def is_ok(res: Result[T]) -> TypeGuard[Result[T]]:
    return not isinstance(res._error, FakeNone) and not isinstance(res._value, FakeNone)

def is_err(res: Result[T]) -> TypeGuard[Result[T]]:
    return not isinstance(res._error, FakeNone) and isinstance(res._value, FakeNone)

def Ok(value: T) -> Result[T]:
    return Result(value, FakeNone())

def Err(error: BaseException) -> Result[T]:
    return Result(FakeNone(), error)